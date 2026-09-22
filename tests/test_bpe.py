"""Small-vocabulary preparation and saved-model inference stay compatible."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from chat import Tokenizer, stream_text
from data.shakespeare.prepare_bpe import prepare
from model import GPT, GPTConfig


ROOT = Path(__file__).resolve().parents[1]


class BPETests(unittest.TestCase):
    def test_prepare_round_trip_and_saved_model_inference(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.txt'
            content = ('ROMEO: The quick brown fox says hello to the world.\n' * 100) + 'café🙂'
            source.write_text(content, encoding='utf-8')
            data = root / 'data'
            metadata = prepare(source, data, vocab_size=280)
            with (data / 'meta.pkl').open('rb') as stream:
                tokenizer = Tokenizer(pickle.load(stream), 280)
            boundary = int(len(content) * 0.9)
            for split, text in [('train', content[:boundary]), ('val', content[boundary:])]:
                ids = np.fromfile(data / f'{split}.bin', dtype=np.uint16).tolist()
                self.assertEqual(ids, tokenizer.encode(text))
                self.assertEqual(''.join(stream_text(ids, tokenizer, stops=())), text)
                self.assertEqual(metadata[split]['tokens'], len(ids))
                self.assertLess(max(ids), 280)
            self.assertEqual(metadata['tokenizer_training_split'], 'train only')
            self.assertEqual(metadata, json.loads((data / 'meta.json').read_text()))
            self.assertTrue(any(len(piece) > 1 for piece in tokenizer.byte_table))
            text = 'Unseen Unicode: 你好, café🙂\n'
            self.assertEqual(''.join(stream_text(tokenizer.encode(text), tokenizer, stops=())), text)

            config = GPTConfig(n_layer=1, n_head=2, n_embd=8, block_size=64, vocab_size=280)
            model = GPT(config)
            torch.save({'model_args': asdict(config), 'model': model.state_dict(),
                        'config': {'dataset': str(data)}, 'iter_num': 0}, root / 'ckpt.pt')
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            for script, extra, input_text in [
                ('sample.py', ['--num_samples=1', '--start=ROMEO:'], None),
                ('chat.py', ['--raw'], 'ROMEO:\n/exit\n'),
            ]:
                result = subprocess.run(
                    [sys.executable, script, f'--out_dir={root}', '--device=cpu',
                     '--dtype=float32', '--max_new_tokens=4', *extra],
                    input=input_text, text=True, capture_output=True, cwd=ROOT, env=env, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('---------------' if script == 'sample.py' else 'Goodbye.', result.stdout)


if __name__ == '__main__':
    unittest.main()
