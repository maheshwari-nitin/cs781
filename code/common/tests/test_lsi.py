"""CPU integration tests using tiny versions of the existing A/B architectures, never submission weights."""
import json
import tempfile
import unittest
import zipfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from lm_lib.config import TrainConfig
from lm_lib.lsi import data
from lm_lib.lsi.metrics import calibrate, decode, scores
from lm_lib.lsi.trainer import LSIConfig, build_classifier, load_lsi_run, predict_test, train_lsi
from lm_lib.utils import build, load_arch


class LSITests(unittest.TestCase):
    def test_metrics_and_calibration(self):
        gold = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]])
        pred = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
        s = scores(gold, pred, [1, 7, 23])
        self.assertAlmostEqual(s['macro_f1'], 1 / 3)
        self.assertAlmostEqual(s['micro_f1'], 4 / 7)
        self.assertEqual(s['subset_accuracy'], 0)
        probs = np.array([[.4, .4, .1], [.4, .1, .1], [.1, .1, .4]])
        for mode in ('global', 'per_label'):
            t = calibrate(gold, probs, mode, min_support=1)
            self.assertEqual(scores(gold, probs >= t)['macro_f1'], 1)
            self.assertEqual(decode(probs, [1, 7, 23], t), [[1, 7], [1], [23]])
        self.assertEqual(decode(np.zeros((1, 3)), [1, 7, 23], .5), [[]])
        # Dev-absent label retains global calibration rather than fitting noise.
        rare_gold = np.array([[1, 0], [0, 0]])
        rare_probs = np.array([[.4, .9], [.1, .8]])
        global_t = calibrate(rare_gold, rare_probs, 'global')
        per_t = calibrate(rare_gold, rare_probs, 'per_label', min_support=1)
        self.assertEqual(global_t[1], per_t[1])

    def test_schema_guards_and_test_label_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.json'
            path.write_text(json.dumps([dict(id='x', text='case', labels=['invalid test sentinel'])]))
            self.assertEqual(data.load_split(path, labeled=False), [dict(id='x', text='case')])
            with self.assertRaises(ValueError):
                data.load_split(path)
            for labels in ([True], [1.0]):
                path.write_text(json.dumps([dict(id='x', text='case', labels=labels)]))
                with self.assertRaises(ValueError):
                    data.load_split(path)
            path.write_text(json.dumps([dict(id='17230589', text='case', labels=[7, 1, 7, 1])]))
            self.assertEqual(data.load_split(path), [dict(id='17230589', text='case', labels=[1, 7])])
            path.write_text(json.dumps([dict(id='x', text='a'), dict(id='x', text='b')]))
            with self.assertRaises(ValueError):
                data.load_split(path, labeled=False)
        with self.assertRaises(ValueError):
            data.check_disjoint(train=[dict(id='same')], dev=[dict(id='same')])

    def test_train_reload_kd_and_submission(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp, patch('torch.cuda.is_available', return_value=False):
            root = Path(tmp)
            vocab = {'<|endoftext|>': 0, '<|pad|>': 1, '<|bos|>': 2, '<unk>': 3,
                     'law': 4, 'case': 5, 'appeal': 6, 'statute': 7}
            tok = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
            tok.pre_tokenizer = pre_tokenizers.Whitespace()
            def checkpoint(name, arch, d, layers):
                path = root / name
                path.mkdir()
                extra = dict(d_model=d, n_heads=2, n_kv_heads=1, head_dim=d // 2, ffn_dim=d * 2,
                             **(dict(n_layers=layers) if arch.startswith('A') else dict(n_enc_layers=layers, n_dec_layers=layers)))
                p1cfg = TrainConfig(arch=arch, seq_len=16, src_len=12, extra=extra)
                model = build(load_arch(arch), len(vocab), 16, extra)
                torch.save(dict(model=model.state_dict(), config=asdict(p1cfg)), path / 'best.pt')
                tok.save(str(path / 'tokenizer.json'))
                return str(path)
            big = checkpoint('big', 'A_qwen_dense', 16, 2)
            small = checkpoint('small', 'B_encdec', 8, 1)
            ds = root / 'data'
            ds.mkdir()
            train = [dict(id=f't{i}', text=('law case appeal statute ' * (i + 1)), labels=lab)
                     for i, lab in enumerate(([1, 7, 7], [7], [23], [1, 23], [1]))]
            dev = [dict(id='d0', text='law case', labels=[1, 81]),
                   dict(id='d1', text='appeal statute law', labels=[7, 23])]
            test = [dict(id='z0', text='law case', labels='must never be accessed'),
                    dict(id='z1', text='', labels=[999999])]
            for split, docs in (('train', train), ('dev', dev), ('test', test)):
                (ds / f'{split}.json').write_text(json.dumps(docs))
            common = dict(data_dir=str(ds), out_dir=str(root / 'runs'), chunk_len=8, max_chunks=2,
                          batch_size=2, grad_accum=2, epochs=1, dropout=0, eval_every=1)
            cfg = LSIConfig(phase1_run=big, run_name='teacher', threshold_mode='per_label',
                            pos_weight='balanced', **common)
            result = train_lsi(cfg)
            self.assertEqual(result['optimizer_steps'], 2)  # includes final partial accumulation group
            self.assertEqual(result['train_unseen_labels'], [81])
            run = root / 'runs' / 'teacher'
            clf, _, _, ids, thresholds, _ = load_lsi_run(str(run), 'cpu')
            self.assertEqual(ids, [1, 7, 23, 81])
            self.assertTrue(np.isfinite(thresholds).all())
            self.assertFalse(clf.training)
            zip_path, predictions = predict_test(str(run), device='cpu')
            self.assertEqual([p['id'] for p in predictions], ['z0', 'z1'])
            self.assertTrue(all(set(p['pred']) <= set(ids) for p in predictions))
            with zipfile.ZipFile(zip_path) as z:
                self.assertEqual(z.namelist(), ['prediction.json'])
                self.assertEqual(json.loads(z.read('prediction.json')), predictions)
            with self.assertRaises(FileExistsError):
                train_lsi(cfg)
            # B encoder paths, smaller-student Bernoulli KD and saved teacher compatibility.
            kd = LSIConfig(phase1_run=small, run_name='student', kd_teacher=str(run), **common)
            student = train_lsi(kd)
            self.assertLess(student['total_params'], result['total_params'])
            predict_test(str(root / 'runs' / 'student'), device='cpu')
            # Existing LoRMA and partial freezing must train/reload through the same interface.
            for name, method, kwargs in [('lorma', 'lorma', {'r': 2}), ('partial', 'partial', {'k': 1})]:
                adapt = LSIConfig(phase1_run=small, run_name=name, method=method, method_kwargs=kwargs, **common)
                out = train_lsi(adapt)
                self.assertGreater(out['trainable_params'], 0)
                load_lsi_run(str(root / 'runs' / name), 'cpu')
            bad = LSIConfig(phase1_run=small, run_name='bad', chunk_len=13)
            with self.assertRaisesRegex(ValueError, 'context'):
                build_classifier(bad, ids, 'cpu')
            for override, error in [({'chunk_len': 4}, 'preprocessing'), ({'phase1_run': big}, 'smaller student')]:
                args = dict(common, phase1_run=small, run_name='bad_kd', kd_teacher=str(run))
                args.update(override)
                with self.assertRaisesRegex(ValueError, error):
                    train_lsi(LSIConfig(**args))


if __name__ == '__main__':
    unittest.main()
