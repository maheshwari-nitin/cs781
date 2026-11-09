"""CPU retrieval integration tests on tiny existing A/B models, with isolated split candidate pools."""
import json
import tempfile
import unittest
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from lm_lib.config import TrainConfig
from lm_lib.pcr import data
from lm_lib.pcr.metrics import retrieval_scores, validate_predictions
from lm_lib.pcr.trainer import (PCRConfig, build_retriever, case_vector, chunks, encode_corpus,
                                embed_token_batch, load_pcr_run, predict_test, rank_embeddings,
                                ranking_batch_losses, train_pcr)
from lm_lib.pcr.batching import calibrate_batches, largest_fitting, training_choices
from lm_lib.pcr.mining import mine_tfidf_negatives
from lm_lib.pcr.tuning import a_more_training_config, select_best_a_run
from lm_lib.utils import build, load_arch


class PCRTests(unittest.TestCase):
    def test_hard_negative_sampling_and_more_training_recipe(self):
        queries = [dict(id='q', text='contract appeal court', relevant_candidates=['p'])]
        pool = [dict(id='p', text='contract appeal court'), dict(id='hard', text='contract appeal court order'),
                dict(id='easy', text='ocean beach sand'), dict(id='other', text='summer holiday')]
        mined = mine_tfidf_negatives(queries, pool, top_k=2)
        self.assertEqual(mined['q'][0], 'hard')
        self.assertNotIn('p', mined['q'])
        pairs, _ = data.sample_pairs(queries, pool, negatives=2, hard_candidates={'q': ['p', 'hard', 'foreign']},
                                     hard_fraction=1)
        self.assertEqual(pairs[0]['n_hard_negative'], 1)
        self.assertEqual(pairs[0]['candidate_ids'][0], 'p')
        self.assertIn('hard', pairs[0]['candidate_ids'][1:])
        self.assertNotIn('foreign', pairs[0]['candidate_ids'])
        base = PCRConfig(run_name='original', train_batch_size=32, grad_accum=1, epochs=3)
        tuned = a_more_training_config(base)
        self.assertEqual(tuned.epochs, 30)
        self.assertEqual(tuned.train_batch_size * tuned.grad_accum, 32)
        self.assertEqual(tuned.chunk_len * tuned.max_chunks, base.chunk_len * base.max_chunks)
        self.assertEqual(base.epochs, 3)
        with self.assertRaisesRegex(ValueError, 'fixed pairs'):
            replace(tuned, kd_teacher='teacher')

    def test_tuning_keeps_baseline_when_candidates_regress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = dict(phase1_sha256='test-base-hash', train_fingerprint='test-train',
                            dev_fingerprint='test-dev', eval_query_ids=['dev'], used_train_query_ids=['train'])
            for name, score in [('baseline', .2), ('worse', .1), ('better', .3)]:
                folder = root / name; folder.mkdir()
                cfg = PCRConfig(run_name=name, phase1_run='test-A')
                (folder / 'config.json').write_text(json.dumps(asdict(cfg)))
                (folder / 'manifest.json').write_text(json.dumps(manifest))
                (folder / 'results.json').write_text(json.dumps(dict(best_dev_micro_f1_at7=score)))
                (folder / 'best.pt').write_bytes(b'test-placeholder')
            report = select_best_a_run(root / 'baseline', [root / 'worse'])
            self.assertEqual(report['selected_run'], str(root / 'baseline'))
            self.assertEqual(report['delta'], 0)
            report = select_best_a_run(root / 'baseline', [root / 'worse', root / 'better'])
            self.assertEqual(report['selected_run'], str(root / 'better'))
            (root / 'better' / 'manifest.json').write_text(json.dumps(dict(manifest, eval_query_ids=['different'])))
            with self.assertRaisesRegex(ValueError, 'identical'):
                select_best_a_run(root / 'baseline', [root / 'better'])

    def test_batch_search_and_cpu_calibration(self):
        self.assertEqual(training_choices(32, 20), [1, 2, 4, 8, 16])
        self.assertEqual(training_choices(30, 32), [1, 2, 3, 5, 6, 10, 15, 30])
        self.assertEqual(largest_fitting(range(1, 101), lambda n: n <= 37), 37)
        self.assertEqual(largest_fitting([1, 2, 4, 8], lambda n: True), 8)
        self.assertEqual(largest_fitting([1], lambda n: True), 1)
        with self.assertRaisesRegex(RuntimeError, 'Even batch size 1'):
            largest_fitting([1, 2, 4], lambda n: False)
        cfg = PCRConfig(run_name='cpu', auto_batch=True, train_batch_size=2, grad_accum=4)
        with patch('lm_lib.pcr.trainer.build_retriever') as builder:
            resolved = calibrate_batches(cfg, device='cpu')
            builder.assert_not_called()
        self.assertFalse(resolved.auto_batch)
        self.assertEqual(resolved.train_batch_size * resolved.grad_accum, 8)
        self.assertEqual(resolved.batch_calibration['status'], 'skipped_no_cuda')
        self.assertTrue(cfg.auto_batch)  # resolving does not mutate the caller's configuration

    def test_official_sentence_schema_and_id_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'test.json').write_text(json.dumps({'0000021652': {'text': ['first', 'second'],
                                                        'relevant_candidates': 'private sentinel'}}))
            (root / 'test_candidates.json').write_text(json.dumps({'0000590455': {'text': ['law', 'case']}}))
            queries, pool = data.load_split(root, 'test', labeled=False)
            self.assertEqual(queries, [dict(id='0000021652', text='first\nsecond')])
            self.assertEqual(pool, [dict(id='0000590455', text='law\ncase')])
            (root / 'dev.json').write_text(json.dumps({'q': {'text': ['a'], 'relevant_candidates': ['wrong-pool']}}))
            (root / 'dev_candidates.json').write_text(json.dumps({'correct': {'text': ['b']}}))
            with self.assertRaisesRegex(ValueError, 'outside its pool'):
                data.load_split(root, 'dev')
        q = [dict(id='q', relevant_candidates=['p1', 'p2', 'p3'])]
        pool = [dict(id=v) for v in ['p1', 'p2', 'p3', 'n1', 'n2']]
        pairs, _ = data.sample_pairs(q, pool, max_positives=1, negatives=2)
        self.assertEqual(set(pairs[0]['candidate_ids'][1:]), {'n1', 'n2'})

    def test_metrics_complete_gold_and_deterministic_ranking(self):
        queries = [dict(id='q', relevant_candidates=[str(i) for i in range(11)])]
        predictions = {'q': [str(i) for i in range(7)]}
        metrics = retrieval_scores(predictions, queries)
        self.assertEqual(metrics['micro_precision_at7'], 1)
        self.assertAlmostEqual(metrics['micro_recall_at7'], 7 / 11)
        self.assertAlmostEqual(metrics['micro_f1_at7'], 14 / 18)
        ids = ['0008', '0001', '0007', '0003', '0002', '0006', '0005', '0004']
        out = rank_embeddings(['q'], torch.ones(1, 2), ids, torch.ones(8, 2))
        self.assertEqual(out['q'], sorted(ids)[:7])
        with self.assertRaises(ValueError):
            validate_predictions({'q': ['0001'] * 7}, ['q'], ids)
        with self.assertRaises(ValueError):
            validate_predictions({'q': ['different'] + sorted(ids)[:6]}, ['q'], ids)

    def test_training_checkpoint_kd_and_test_pool(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp, patch('torch.cuda.is_available', return_value=False):
            root = Path(tmp)
            vocab = {'<|endoftext|>': 0, '<|pad|>': 1, '<|bos|>': 2, '<unk>': 3,
                     'law': 4, 'case': 5, 'appeal': 6, 'statute': 7}
            tok = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
            tok.pre_tokenizer = pre_tokenizers.Whitespace()
            def checkpoint(name, arch, dim, layers):
                path = root / name
                path.mkdir()
                extra = dict(d_model=dim, n_heads=2, n_kv_heads=1, head_dim=dim // 2, ffn_dim=dim * 2,
                             **(dict(n_layers=layers) if arch.startswith('A') else dict(n_enc_layers=layers, n_dec_layers=layers)))
                cfg = TrainConfig(arch=arch, seq_len=24, src_len=16, extra=extra)
                model = build(load_arch(arch), tok.get_vocab_size(), 24, extra)
                torch.save(dict(model=model.state_dict(), config=asdict(cfg)), path / 'best.pt')
                tok.save(str(path / 'tokenizer.json'))
                return str(path)
            big = checkpoint('big', 'A_qwen_dense', 16, 2)
            small = checkpoint('small', 'B_encdec', 8, 1)
            ds = root / 'data'
            ds.mkdir()
            for split, prefix in [('train', 't'), ('dev', 'd'), ('test', 'z')]:
                pool = {f'{prefix}{i:04}': {'text': ['law case ' * (i + 1), 'appeal statute']} for i in range(9)}
                pool[f'{prefix}0008']['text'] = []  # blank cases keep their IDs and a BOS representation
                queries = {f'{prefix}q{i}': dict(text=['law appeal', 'case ' * (i + 1)],
                            relevant_candidates=[f'{prefix}{i:04}', f'{prefix}{i + 1:04}']) for i in range(3)}
                if split == 'test':
                    for q in queries.values():
                        q['relevant_candidates'] = 'must never be read'
                (ds / f'{split}.json').write_text(json.dumps(queries))
                (ds / f'{split}_candidates.json').write_text(json.dumps(pool))
            common = dict(data_dir=str(ds), out_dir=str(root / 'runs'), chunk_len=8, max_chunks=2,
                          negatives=2, grad_accum=2, epochs=1, embed_batch_size=3, grad_checkpoint=True)
            cfg = PCRConfig(phase1_run=big, method='full', run_name='teacher', **common)
            # Exercise the CUDA calibration control flow on CPU with measured-memory stubs.
            # Real CUDA capacity/performance is intentionally left to the notebook's live probes.
            import lm_lib.pcr.trainer as pt
            from lm_lib.pcr import batching as pb
            original_build = pt.build_retriever
            original_losses = pt.ranking_batch_losses
            original_embed = pt.embed_token_batch
            measured = {'bytes': 100 * 2**20}
            def cpu_probe_build(conf, _device):
                return original_build(conf, 'cpu')
            def cpu_probe_loss(model_arg, pairs_arg, qtok, ctok, conf, _device, kd_arg, ttemp):
                losses = original_losses(model_arg, pairs_arg, qtok, ctok, conf, 'cpu', kd_arg, ttemp)
                measured['bytes'] = (100 + 200 * len(pairs_arg)) * 2**20
                return losses
            def cpu_probe_embed(model_arg, rows, conf, _device):
                measured['bytes'] = (100 + 100 * len(rows)) * 2**20
                return original_embed(model_arg, rows, conf, 'cpu')
            probe_cfg = replace(cfg, effective_batch_queries=8, max_train_batch=8, max_embed_batch=8,
                                grad_checkpoint=False)
            rng_before = torch.get_rng_state().clone()
            with patch.multiple('torch.cuda',
                    get_rng_state_all=lambda: [], set_rng_state_all=lambda _: None,
                    empty_cache=lambda: None, synchronize=lambda: None,
                    reset_peak_memory_stats=lambda: None,
                    mem_get_info=lambda: (512 * 2**20, 1024 * 2**20),
                    memory_reserved=lambda: 128 * 2**20,
                    max_memory_reserved=lambda: measured['bytes'],
                    max_memory_allocated=lambda: measured['bytes'],
                    get_device_name=lambda: 'simulated GPU'), \
                    patch.object(pt, 'build_retriever', side_effect=cpu_probe_build), \
                    patch.object(pt, 'ranking_batch_losses', side_effect=cpu_probe_loss), \
                    patch.object(pt, 'embed_token_batch', side_effect=cpu_probe_embed), \
                    patch.object(pb, 'grad_scaler', side_effect=lambda _: torch.amp.GradScaler('cuda', enabled=False)):
                calibrated = calibrate_batches(probe_cfg, device='cuda', candidate_count=3)
            self.assertEqual(calibrated.train_batch_size, 2)
            self.assertEqual(calibrated.grad_accum, 4)
            self.assertEqual(calibrated.embed_batch_size, 4)
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
            self.assertEqual(probe_cfg.train_batch_size, 1)
            model, tok2, _ = build_retriever(cfg, 'cpu')
            _, docs = data.load_split(ds, 'dev')
            model.eval()
            with torch.no_grad():
                single = torch.stack([case_vector(model, chunks(d, tok2, cfg), cfg, 'cpu') for d in docs])
            batched = encode_corpus(model, tok2, docs, cfg, 'cpu')
            torch.testing.assert_close(single, batched, atol=1e-5, rtol=1e-5)
            # Physical query batching preserves each query's separate ranking objective AND its gradients.
            train_queries, train_pool = data.load_split(ds, 'train')
            pairs, _ = data.sample_pairs(train_queries, train_pool, cfg.negatives, cfg.max_positives, cfg.seed)
            qtokens = {d['id']: chunks(d, tok2, cfg) for d in train_queries}
            ctokens = {d['id']: chunks(d, tok2, cfg) for d in train_pool}
            micro_cfg = replace(cfg, train_chunk_batch_size=3)
            model.zero_grad(set_to_none=True)
            sequential = torch.stack([ranking_batch_losses(model, [p], qtokens, ctokens,
                                                            micro_cfg, 'cpu')[0] for p in pairs[:2]]).mean()
            sequential.backward()
            expected_grad = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            combined = ranking_batch_losses(model, pairs[:2], qtokens, ctokens, micro_cfg, 'cpu').mean()
            combined.backward()
            torch.testing.assert_close(sequential.detach(), combined.detach(), atol=1e-5, rtol=1e-5)
            for name, p in model.named_parameters():
                if name in expected_grad:
                    torch.testing.assert_close(p.grad, expected_grad[name], atol=3e-5, rtol=3e-4)
            model.zero_grad(set_to_none=True)
            # Simulate a CUDA OOM on large embedding batches: retries must preserve every document.
            retry_cfg = replace(cfg, embed_batch_size=5)
            def limited_forward(model_arg, rows, conf, _device):
                if len(rows) > 2:
                    raise torch.cuda.OutOfMemoryError('simulated embedding batch limit')
                return embed_token_batch(model_arg, rows, conf, 'cpu')
            with patch('lm_lib.pcr.trainer.embed_token_batch', side_effect=limited_forward), \
                    patch('torch.cuda.empty_cache'):
                retried = encode_corpus(model, tok2, docs, retry_cfg, 'cuda')
            self.assertLessEqual(retry_cfg.embed_batch_size, 2)
            torch.testing.assert_close(retried, batched, atol=1e-5, rtol=1e-5)
            result = train_pcr(cfg)
            self.assertEqual(result['optimizer_steps'], 2)  # last accumulation group has one query
            run = root / 'runs' / 'teacher'
            path, predictions = predict_test(str(run), device='cpu')
            validate_predictions(predictions, [f'zq{i}' for i in range(3)], [f'z{i:04}' for i in range(9)])
            self.assertTrue(all(v.startswith('z') for values in predictions.values() for v in values))
            with zipfile.ZipFile(path) as z:
                self.assertEqual(z.namelist(), ['prediction.json'])
                self.assertEqual(json.loads(z.read('prediction.json')), predictions)
            # Test-pool cache can be reused only for the same learned checkpoint and pool fingerprint.
            _, again = predict_test(str(run), device='cpu')
            self.assertEqual(predictions, again)
            with self.assertRaises(FileExistsError):
                train_pcr(cfg)
            student = PCRConfig(phase1_run=small, method='lorma', method_kwargs={'r': 2},
                                run_name='student', kd_teacher=str(run), **common)
            trained = train_pcr(student)
            self.assertLess(trained['base_params'], result['base_params'])
            student_run = root / 'runs' / 'student'
            targets = json.loads((student_run / 'kd_targets.json').read_text())['targets']
            self.assertEqual(set(targets), {f'tq{i}' for i in range(3)})
            self.assertTrue(all(v.startswith('t') for row in targets.values() for v in row['candidate_ids']))
            loaded, _, _, _ = load_pcr_run(str(student_run), 'cpu')
            self.assertTrue(all(not p.requires_grad for p in loaded.decoder.parameters()))
            predict_test(str(student_run), device='cpu')
            partial = PCRConfig(phase1_run=small, method='partial', method_kwargs={'k': 1},
                                run_name='partial', **dict(common, train_batch_size=2, grad_accum=1))
            partial_result = train_pcr(partial)
            self.assertEqual(partial_result['optimizer_steps'], 2)  # also handles a short physical microbatch
            self.assertEqual(partial_result['effective_batch_queries'], 2)
            load_pcr_run(str(root / 'runs' / 'partial'), 'cpu')
            # Early stopping uses dev metrics; fresh TRAIN samples are saved for every completed epoch.
            resampled = PCRConfig(phase1_run=big, method='full', run_name='resampled',
                                  negative_sampling='tfidf', resample_negatives=True, early_stopping_patience=2,
                                  **dict(common, epochs=10))
            fixed_dev = {f'dq{i}': [f'd{j:04}' for j in range(7)] for i in range(3)}
            with patch('lm_lib.pcr.trainer.retrieve', return_value=fixed_dev):
                stopped = train_pcr(resampled)
            self.assertEqual(stopped['completed_epochs'], 3)
            self.assertEqual(stopped['stop_reason'], 'dev_patience')
            pair_files = list((root / 'runs' / 'resampled').glob('train_pairs_epoch_*.json'))
            self.assertEqual(len(pair_files), 3)
            for pair_file in pair_files:
                for pair in json.loads(pair_file.read_text()):
                    self.assertTrue(all(v.startswith('t') for v in pair['candidate_ids']))
                    query = next(q for q in train_queries if q['id'] == pair['query_id'])
                    self.assertFalse(set(pair['candidate_ids'][pair['n_positive']:]) & set(query['relevant_candidates']))
            broken = torch.load(student_run / 'best.pt', weights_only=False)
            broken['state'].pop(next(iter(broken['state'])))
            torch.save(broken, student_run / 'best.pt')
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                load_pcr_run(str(student_run), 'cpu')


if __name__ == '__main__':
    unittest.main()
