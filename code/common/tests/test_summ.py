"""CPU regressions for Phase 6 using tiny existing A/B architectures; no competition data."""
import json
import tempfile
import unittest
import zipfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from lm_lib.config import TrainConfig
from lm_lib.summ import data
from lm_lib.summ.trainer import (SummConfig, build_model, generate_docs, generate_one, load_summ_run,
                                  predict_test, summary_loss, train_summ)
from lm_lib.utils import build, load_arch


def tokenizer():
    vocab = {'<|endoftext|>': 0, '<|pad|>': 1, '<|bos|>': 2, '<unk>': 3,
             'law': 4, 'case': 5, 'appeal': 6, 'statute': 7, 'Legal': 8,
             'document': 9, ':': 10, 'Summary': 11}
    tok = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


class SummTests(unittest.TestCase):
    def test_missing_training_summaries_are_reported_and_test_ids_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'train.json'
            records = [dict(id='ok', document='case', summary='law'),
                       dict(id='299', document='case two', summary='   '),
                       dict(id='null', document='case three', summary=None),
                       dict(id='missing', document='case four')]
            p.write_text(json.dumps(records))
            quality = {}
            with self.assertWarnsRegex(UserWarning, 'skipped 3'):
                loaded = data.load_split(p, report=quality)
            self.assertEqual(loaded, [records[0]])
            self.assertEqual(quality['skipped_empty_summary_ids'], ['299', 'null', 'missing'])
            self.assertEqual(quality['total_records'], 4)
            self.assertEqual(quality['loaded_records'], 1)
            with self.assertRaisesRegex(ValueError, '299'):
                data.load_split(p, skip_empty_summaries=False)
            # Inference must retain every ID regardless of reference quality.
            self.assertEqual([d['id'] for d in data.load_split(p, labeled=False)],
                             ['ok', '299', 'null', 'missing'])
            p.write_text(json.dumps(records[1:]))
            with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, 'No usable records'):
                data.load_split(p)

    def test_schema_holdout_and_token_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'test.json'
            p.write_text(json.dumps([dict(id='x', document='case', summary=['private reference sentinel'])]))
            self.assertEqual(data.load_split(p, labeled=False), [dict(id='x', document='case')])
            with self.assertRaises(ValueError):
                data.load_split(p)
        docs = [dict(id=str(i), document=s, summary='law')
                for i, s in enumerate(['a case', 'a  case', 'another', 'third', 'fourth'])]
        train, val = data.split_train_val(docs, .4, 12)
        self.assertFalse({tuple(d['document'].split()) for d in train} &
                         {tuple(d['document'].split()) for d in val})
        self.assertEqual(data.split_train_val(list(reversed(docs)), .4, 12), (train, val))
        cfg = SummConfig(run_name='test', source_tokens=12, target_tokens=5, max_new_tokens=5, min_new_tokens=1)
        tok = tokenizer()
        doc = dict(id='d', document='law case ' * 20, summary='appeal statute law case appeal statute')
        prefix = data.source_ids(doc, tok, cfg)
        ex = data.prepare(doc, tok, cfg, 'decoder')
        self.assertLessEqual(len(prefix), cfg.source_tokens)
        self.assertEqual(ex['y'][:len(prefix) - 1], [-100] * (len(prefix) - 1))
        self.assertEqual(ex['y'][len(prefix) - 1:], [6, 7, 4, 5, 0])
        self.assertEqual(len(ex['x']), len(ex['y']))
        enc = data.prepare(doc, tok, cfg, 'encdec')
        self.assertEqual(enc['x'], [2, 6, 7, 4, 5])
        self.assertEqual(enc['y'], [6, 7, 4, 5, 0])
        self.assertEqual(data.select_tokens(list(range(10)), 5, 'first_last'), [0, 1, 2, 8, 9])

    def test_eos_stops_cached_generation(self):
        tok = tokenizer()
        cfg = SummConfig(run_name='eos', source_tokens=12, target_tokens=5, max_new_tokens=5, min_new_tokens=1)
        class Scripted:
            def eval(self):
                return self
            def generate_tokens(self, idx, max_new, sample_fn):
                self.calls = 0
                for v in [4, 0, 6, 7, 5]:
                    self.calls += 1
                    logits = torch.full((1, 12), -10.)
                    logits[0, v] = 10
                    sample_fn(logits, idx)
        model = Scripted()
        doc = dict(id='d', document='law case')
        self.assertEqual(generate_one(model, tok, 'decoder', doc, cfg, 'cpu'), 'law')
        self.assertEqual(model.calls, 2)
        before = torch.get_rng_state().clone()
        self.assertEqual(generate_docs(model, tok, 'decoder', [doc], cfg, 'cpu'), {'d': 'law'})
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_training_adapters_kd_and_export(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp, patch('torch.cuda.is_available', return_value=False):
            root = Path(tmp)
            tok = tokenizer()
            def phase1(name, arch, dim, layers):
                path = root / name
                path.mkdir()
                extra = dict(d_model=dim, n_heads=2, n_kv_heads=1, head_dim=dim // 2, ffn_dim=dim * 2,
                             **(dict(n_layers=layers) if arch.startswith('A') else dict(n_enc_layers=layers, n_dec_layers=layers)))
                cfg = TrainConfig(arch=arch, seq_len=32, src_len=20, extra=extra)
                model = build(load_arch(arch), tok.get_vocab_size(), 32, extra)
                torch.save(dict(model=model.state_dict(), config=asdict(cfg)), path / 'best.pt')
                tok.save(str(path / 'tokenizer.json'))
                return str(path)
            big = phase1('big', 'A_qwen_dense', 16, 2)
            small = phase1('small', 'B_encdec', 8, 1)
            ds = root / 'data'
            ds.mkdir()
            docs = [dict(id=f't{i}', document='law case appeal ' * (i + 1), summary='law appeal case') for i in range(6)]
            (ds / 'train.json').write_text(json.dumps(docs))
            (ds / 'test.json').write_text(json.dumps([dict(id='z0', document='appeal case', summary=['private'])]))
            common = dict(data_dir=str(ds), out_dir=str(root / 'runs'), val_fraction=.33, source_tokens=12,
                          target_tokens=5, max_new_tokens=5, min_new_tokens=1, grad_accum=3,
                          epochs=1, n_val_generate=1, selection_metric='rougeL_f1')
            cfg = SummConfig(phase1_run=big, method='lora', method_kwargs={'r': 2}, run_name='teacher', **common)
            # Adapter gradients on causal summary-only targets; frozen tied embeddings stay frozen.
            model, tok2, mtype, _ = build_model(cfg, 'cpu')
            self.assertFalse(model.tok_embeddings.weight.requires_grad)
            ex = data.prepare(docs[0], tok2, cfg, mtype)
            loss, ntokens = summary_loss(model, ex, mtype, 'cpu')
            loss.backward()
            self.assertEqual(ntokens, 4)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for n, p in model.named_parameters() if 'lora_B' in n))
            result = train_summ(cfg)
            self.assertEqual(result['optimizer_steps'], 2)
            teacher_run = root / 'runs' / 'teacher'
            checkpoint = torch.load(teacher_run / 'best.pt', weights_only=False)
            self.assertTrue(checkpoint['partial'])
            self.assertFalse(any('base_layer.weight' in n for n in checkpoint['state']))
            model1, tok1, cfg1, mt1, _ = load_summ_run(str(teacher_run), 'cpu')
            model2, tok2, cfg2, mt2, _ = load_summ_run(str(teacher_run), 'cpu')
            self.assertEqual(generate_one(model1, tok1, mt1, docs[0], cfg1, 'cpu'),
                             generate_one(model2, tok2, mt2, docs[0], cfg2, 'cpu'))
            broken = dict(checkpoint, state=dict(checkpoint['state']))
            broken['state'].pop(next(iter(broken['state'])))
            torch.save(broken, teacher_run / 'best.pt')
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                load_summ_run(str(teacher_run), 'cpu')
            torch.save(checkpoint, teacher_run / 'best.pt')
            path, records = predict_test(str(teacher_run), device='cpu')
            self.assertEqual([r['id'] for r in records], ['z0'])
            self.assertIsInstance(records[0]['summary'], str)
            self.assertNotIn('pred', records[0])
            with zipfile.ZipFile(path) as z:
                self.assertEqual(z.namelist(), ['prediction.json'])
                self.assertEqual(json.loads(z.read('prediction.json')), records)
            with self.assertRaises(FileExistsError):
                train_summ(cfg)
            # True encoder-decoder sequence KD, with teacher trained on exactly the same holdout split.
            student_cfg = SummConfig(phase1_run=small, method='lorma', method_kwargs={'r': 2},
                                     kd_teacher=str(teacher_run), run_name='student', grad_checkpoint=True, **common)
            encoder_decoder, stok, smtype, _ = build_model(student_cfg, 'cpu')
            sloss, _ = summary_loss(encoder_decoder, data.prepare(docs[0], stok, student_cfg, smtype), smtype, 'cpu')
            sloss.backward()
            self.assertTrue(any(p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
                                for p in encoder_decoder.decoder.parameters()))
            student_result = train_summ(student_cfg)
            self.assertLess(student_result['base_params'], result['base_params'])
            student_run = root / 'runs' / 'student'
            manifest = json.loads((student_run / 'split.json').read_text())
            targets = json.loads((student_run / 'kd_targets.json').read_text())['predictions']
            self.assertEqual(set(targets), set(manifest['train_ids']))
            self.assertFalse(set(targets) & set(manifest['val_ids']))
            predict_test(str(student_run), device='cpu')
            # Split mismatch is rejected before teacher predictions are constructed.
            bad = SummConfig(phase1_run=small, method='lora', kd_teacher=str(teacher_run),
                             run_name='bad_split', **dict(common, val_fraction=.5))
            with self.assertRaisesRegex(ValueError, 'held-out split'):
                train_summ(bad)
            with self.assertRaisesRegex(ValueError, 'CUDA'):
                build_model(SummConfig(phase1_run=big, method='qlora', run_name='q', **common), 'cpu')
            # Full checkpoint path is also supported, without leaving B's decoder frozen.
            full = SummConfig(phase1_run=small, method='full', run_name='full', **common)
            train_summ(full)
            self.assertFalse(torch.load(root / 'runs/full/best.pt', weights_only=False)['partial'])
            load_summ_run(str(root / 'runs/full'), 'cpu')


if __name__ == '__main__':
    unittest.main()
