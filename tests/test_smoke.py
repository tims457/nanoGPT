"""Small integration checks: local HF datasets, two CPU updates, W&B, and sampling."""

from argparse import Namespace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
import torch


ROOT = Path(__file__).resolve().parents[1]


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'tokens'
        self.source = self.root / 'source.jsonl'
        self.rows = [f'Document {i}. A small language model reads this sentence. ' * 4 for i in range(8)]
        self.source.write_text(''.join(json.dumps({'body': text}) + '\n' for text in self.rows))
        self.env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                        WANDB_MODE='offline', WANDB_CONFIG_DIR=str(self.root / 'wandb-config'),
                        WANDB_CACHE_DIR=str(self.root / 'wandb-cache'))
        self.env.pop('WANDB_RUN_ID', None)
        self.env.pop('WANDB_RESUME', None)
        # A tokenizer download is needed only on the first invocation.
        self.env.setdefault('TIKTOKEN_CACHE_DIR', str(ROOT / '.cache' / 'tiktoken'))

    def run_script(self, *args, success=True, capture_wandb=False, wandb_server_step=-1):
        command = [sys.executable, *map(str, args)]
        if capture_wandb:
            # Capture exact API values while still writing a real offline W&B run.
            wrapper = '''
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
import wandb
from wandb.sdk.wandb_run import Run

capture_path = Path(sys.argv[1])
server_step = int(sys.argv[2])
sys.argv = sys.argv[3:]
original_init, original_log = wandb.init, Run.log

def init_local(**kwargs):
    # Exercise online resume arguments without contacting W&B or uploading data.
    if server_step >= 0:
        assert kwargs['mode'] == 'online'
        kwargs.update(mode='offline', resume=None)
    run = original_init(**kwargs)
    if server_step > 0:
        run.config.update({'max_iters': 0}, allow_val_change=True)
        original_log(run, {'test/server_history': True}, step=server_step - 1, commit=True)
    return run

with patch('wandb.init', side_effect=init_local) as init, \\
     patch.object(Run, 'log', autospec=True, side_effect=original_log) as log:
    result = runpy.run_path(sys.argv[0], run_name='__main__')
capture_path.write_text(json.dumps({
    'config': init.call_args.kwargs['config'],
    'actual_config': dict(result['wandb_run'].config),
    'project': init.call_args.kwargs['project'],
    'init': {key: value for key, value in init.call_args.kwargs.items() if key != 'config'},
    'history': [call.args[1] for call in log.call_args_list],
    'steps': [call.kwargs['step'] for call in log.call_args_list],
}))
'''
            command = [sys.executable, '-c', wrapper, str(self.root / 'wandb.json'),
                       str(wandb_server_step), *map(str, args)]
        result = subprocess.run(command, cwd=ROOT,
                                env=self.env, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
        if success:
            self.assertEqual(result.returncode, 0, output)
        else:
            self.assertNotEqual(result.returncode, 0, output)
        return output

    def assert_wandb_model_config(self, checkpoint):
        logged = json.loads((self.root / 'wandb.json').read_text())
        self.assertEqual(logged['project'], 'lm_research')
        self.assertEqual(logged['config']['wandb_project'], 'lm_research')
        for key, value in logged['config'].items():
            self.assertEqual(logged['actual_config'][key], value)
        for key, value in checkpoint['model_args'].items():
            self.assertEqual(logged['config'][key], value)
        # The state dict includes tied token/output weights twice; count them once.
        expected_params = sum(value.numel() for key, value in checkpoint['model'].items()
                              if key != 'lm_head.weight')
        self.assertEqual(logged['config']['num_params'], expected_params)
        self.assertEqual(checkpoint['config']['num_params'], expected_params)
        return logged

    def prepare(self, *extra, success=True):
        return self.run_script(
            'data/huggingface/prepare.py', '--dataset=json', '--train-file', self.source,
            '--text-column=body', '--val-split=', '--max-train-examples=3',
            '--max-val-examples=2', '--output-dir', self.data, *extra, success=success,
        )

    def test_bounded_disjoint_holdout_and_failed_prepare_preserves_data(self):
        self.prepare()
        with patch.dict(os.environ, {'TIKTOKEN_CACHE_DIR': self.env['TIKTOKEN_CACHE_DIR']}):
            encoding = tiktoken.get_encoding('gpt2')
        for split, texts in [('val', self.rows[:2]), ('train', self.rows[2:5])]:
            expected = [token for text in texts for token in encoding.encode_ordinary(text) + [encoding.eot_token]]
            actual = np.fromfile(self.data / f'{split}.bin', dtype=np.uint16)
            np.testing.assert_array_equal(actual, expected)
        before = (self.data / 'train.bin').read_bytes()
        failure = self.prepare('--text-column=missing', success=False)
        self.assertIn('Missing text column', failure)
        self.assertEqual((self.data / 'train.bin').read_bytes(), before)
        metadata = json.loads((self.data / 'meta.json').read_text())
        self.assertEqual(metadata['train']['examples'], 3)
        self.assertEqual(metadata['val']['examples'], 2)

    def test_explicit_parquet_validation_split(self):
        source = self.root / 'train.parquet'
        validation = self.root / 'validation.parquet'
        pq.write_table(pa.table({'body': self.rows}), source, row_group_size=2)
        pq.write_table(pa.table({'body': ['Separate validation document. ' * 40] * 5}), validation, row_group_size=2)
        expected = None
        for options in ((), ('--download',)):
            with self.subTest(download=bool(options)):
                self.prepare('--dataset=parquet', '--train-file', source,
                             '--val-split=validation', '--val-file', validation, *options)
                metadata = json.loads((self.data / 'meta.json').read_text())
                self.assertEqual(metadata['val']['examples'], 2)
                self.assertEqual(metadata['train']['examples'], 3)
                self.assertEqual(metadata['data_loading'], 'downloaded' if options else 'streaming')
                actual = [(self.data / f'{split}.bin').read_bytes() for split in ('train', 'val')]
                if expected is None:
                    expected = actual
                self.assertEqual(actual, expected)

    def test_openwebtext_downloads_before_bounded_tokenization(self):
        from datasets import load_dataset_builder
        from data.huggingface import prepare as hf_prepare

        self.prepare()
        expected = [(self.data / f'{split}.bin').read_bytes() for split in ('train', 'val')]
        # Exercise OpenWebText's automatic download path using a small local source.
        builder = load_dataset_builder('json', data_files={'train': str(self.source)})
        args = Namespace(
            dataset=hf_prepare.OPENWEBTEXT, dataset_config=None, revision=None,
            download=False, train_file=None, val_file=None, text_column='body',
            train_split='train', val_split='', max_train_examples=3,
            max_val_examples=2, block_size=32, output_dir=str(self.data),
        )
        with patch.object(hf_prepare, 'load_dataset_builder', return_value=builder) as load, \
             patch.object(hf_prepare, 'stream_rows', side_effect=AssertionError('OpenWebText must not stream')), \
             patch.dict(os.environ, {'TIKTOKEN_CACHE_DIR': self.env['TIKTOKEN_CACHE_DIR']}):
            metadata = hf_prepare.prepare(args)
        load.assert_called_once_with(path=hf_prepare.OPENWEBTEXT, name='default',
                                     revision=hf_prepare.OPENWEBTEXT_REVISION)
        self.assertEqual(metadata['data_loading'], 'downloaded')
        self.assertEqual(metadata['revision'], hf_prepare.OPENWEBTEXT_REVISION)
        self.assertEqual([(self.data / f'{split}.bin').read_bytes() for split in ('train', 'val')], expected)
        # All source rows were cached, including those outside the tokenization limit.
        self.source.unlink()
        cached = builder.as_dataset(split='train', in_memory=False)
        self.assertEqual(len(cached), len(self.rows))
        self.assertEqual(cached[-1]['body'], self.rows[-1])

    def test_train_log_resume_and_sample(self):
        self.prepare()
        out = self.root / 'run'
        args = ['train.py', 'config/smoke.py', f'--dataset={self.data}', f'--out_dir={out}']
        output = self.run_script(*args, capture_wandb=True)
        updates = [line for line in output.splitlines() if line.startswith('iter ')]
        self.assertEqual(len(updates), 2, output)
        self.assertIn('step 2: train loss', output)
        self.assertIn('Run summary:', output)
        for key in ('train/batch_loss', 'train/loss', 'val/loss', 'tokens_per_second',
                    'train/perplexity', 'val/perplexity'):
            self.assertIn(key, output)
        runs = list((out / 'wandb').glob('offline-run-*/run-*.wandb'))
        self.assertEqual(len(runs), 1)
        self.assertGreater(runs[0].stat().st_size, 0)
        checkpoint = torch.load(out / 'ckpt.pt', map_location='cpu', weights_only=True)
        self.assertEqual(checkpoint['iter_num'], 2)
        self.assertTrue(np.isfinite(checkpoint['best_val_loss']))
        self.assertEqual(checkpoint['config']['data']['dataset'], 'json')
        self.assertEqual(checkpoint['config']['data']['train']['examples'], 3)
        logged = self.assert_wandb_model_config(checkpoint)
        self.assertEqual(checkpoint['wandb']['id'], logged['init']['id'])
        self.assertEqual(checkpoint['wandb']['project'], 'lm_research')
        self.assertIsNone(logged['init']['resume']) # offline uses separate local segments
        self.assertEqual(logged['steps'], list(range(5)))
        self.assertEqual(checkpoint['wandb']['step'], 5)
        evaluations = [row for row in logged['history'] if 'val/loss' in row]
        self.assertEqual([row['iter'] for row in evaluations], [0, 1, 2])
        for row in evaluations:
            for split in ('train', 'val'):
                self.assertTrue(math.isclose(row[f'{split}/perplexity'],
                                             math.exp(row[f'{split}/loss']), rel_tol=1e-12))
        optimizer_steps = [int(state['step']) for state in checkpoint['optimizer']['state'].values()]
        self.assertTrue(optimizer_steps)
        self.assertEqual(set(optimizer_steps), {2})

        # Evaluation of a resumed checkpoint must not take another optimizer step,
        # even when the checkpoint is not on the requested evaluation interval.
        # Logged dimensions must come from the loaded/cropped model, not overrides.
        output = self.run_script(*args, '--init_from=resume', '--eval_only=True',
                                 '--eval_interval=5', '--n_layer=9', '--n_head=8',
                                 '--n_embd=64', '--block_size=16', capture_wandb=True)
        self.assertIn('step 2: train loss', output)
        self.assertFalse(any(line.startswith('iter ') for line in output.splitlines()))
        resumed = torch.load(out / 'ckpt.pt', map_location='cpu', weights_only=True)
        resumed_log = self.assert_wandb_model_config(resumed)
        self.assertEqual(resumed['wandb']['id'], checkpoint['wandb']['id'])
        self.assertEqual(resumed_log['init']['id'], checkpoint['wandb']['id'])
        self.assertEqual(resumed_log['steps'], [5])
        self.assertEqual(resumed['wandb']['step'], 6)
        self.assertEqual(resumed['config']['n_layer'], 1)
        self.assertEqual(resumed['config']['n_head'], 2)
        self.assertEqual(resumed['config']['n_embd'], 32)
        self.assertEqual(resumed['config']['block_size'], 16)
        self.assertEqual(resumed['config']['num_params'], checkpoint['config']['num_params'] - 16 * 32)
        output = self.run_script('sample.py', f'--out_dir={out}', '--device=cpu',
                                 '--dtype=float32', '--num_samples=1', '--max_new_tokens=4')
        self.assertIn('---------------', output)

        # A too-short dataset should fail before a model or W&B run is created.
        (self.data / 'val.bin').write_bytes(b'\x00\x00')
        output = self.run_script(*args, '--wandb_log=False', success=False)
        self.assertIn('needs at least 33 tokens', output)

    def test_wandb_online_resume_and_legacy_checkpoint(self):
        self.prepare()
        out = self.root / 'run'
        args = ['train.py', 'config/smoke.py', f'--dataset={self.data}', f'--out_dir={out}',
                '--wandb_mode=online', '--wandb_entity=test-team']
        self.run_script(*args, capture_wandb=True, wandb_server_step=0)
        first = torch.load(out / 'ckpt.pt', map_location='cpu', weights_only=True)
        first_log = self.assert_wandb_model_config(first)
        self.assertEqual(first_log['init']['resume'], 'never')
        self.assertEqual(first['wandb']['entity'], 'test-team')
        run_id = first['wandb']['id']

        # Online history can be ahead of the saved checkpoint; append after it.
        server_step = first['wandb']['step'] + 3
        output = self.run_script(*args, '--init_from=resume', '--max_iters=4',
                                 '--wandb_project=ignored-project', '--wandb_entity=ignored-team',
                                 '--wandb_run_name=ignored-name',
                                 capture_wandb=True, wandb_server_step=server_step)
        resumed = torch.load(out / 'ckpt.pt', map_location='cpu', weights_only=True)
        logged = self.assert_wandb_model_config(resumed)
        self.assertEqual(logged['init']['id'], run_id)
        self.assertEqual(logged['init']['resume'], 'allow')
        self.assertEqual(logged['init']['entity'], 'test-team')
        self.assertEqual(logged['init']['name'], first['wandb']['name'])
        self.assertEqual(logged['steps'], list(range(server_step, server_step + 5)))
        self.assertEqual(resumed['wandb']['step'], server_step + 5)
        self.assertEqual(logged['history'][0]['iter'], 2)
        self.assertEqual(logged['history'][-1]['iter'], 4)
        self.assertEqual(len([line for line in output.splitlines() if line.startswith('iter ')]), 2)
        self.assertEqual(resumed['iter_num'], 4)
        self.assertEqual({int(state['step']) for state in resumed['optimizer']['state'].values()}, {4})

        # Disabling logging temporarily must preserve the saved run association.
        for override in ('--wandb_log=False', '--wandb_mode=disabled'):
            self.run_script(*args, '--init_from=resume', '--eval_only=True', override)
            disabled = torch.load(out / 'ckpt.pt', map_location='cpu', weights_only=True)
            self.assertEqual(disabled['wandb'], resumed['wandb'])

        failure = self.run_script(*args, '--init_from=resume', '--wandb_run_id=wrong-id', success=False)
        self.assertIn('conflicts with the run saved in the checkpoint', failure)

        # Old checkpoints can explicitly attach to an existing run by ID.
        legacy = dict(resumed)
        del legacy['wandb']
        torch.save(legacy, out / 'ckpt.pt')
        self.run_script(*args, '--init_from=resume', '--eval_only=True',
                        f'--wandb_run_id={run_id}', capture_wandb=True,
                        wandb_server_step=server_step + 5)
        attached = torch.load(out / 'ckpt.pt', map_location='cpu', weights_only=True)
        logged = self.assert_wandb_model_config(attached)
        self.assertEqual(attached['wandb']['id'], run_id)
        self.assertEqual(logged['init']['resume'], 'allow')

        # Without an ID, legacy checkpoints still train and establish a new run.
        torch.save(legacy, out / 'ckpt.pt')
        output = self.run_script(*args, '--init_from=resume', '--eval_only=True',
                                 capture_wandb=True, wandb_server_step=0)
        self.assertIn('Checkpoint has no W&B run ID', output)
        new = torch.load(out / 'ckpt.pt', map_location='cpu', weights_only=True)
        self.assertNotEqual(new['wandb']['id'], run_id)
        self.assertEqual(self.assert_wandb_model_config(new)['init']['resume'], 'never')

        # Starting from scratch in the same directory must not resume that run.
        self.run_script(*args, '--max_iters=1', capture_wandb=True, wandb_server_step=0)
        fresh = torch.load(out / 'ckpt.pt', map_location='cpu', weights_only=True)
        self.assertNotEqual(fresh['wandb']['id'], new['wandb']['id'])
        self.assertEqual(self.assert_wandb_model_config(fresh)['init']['resume'], 'never')


if __name__ == '__main__':
    unittest.main()
