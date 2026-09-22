"""Bounded checks for checkpoint loading, streaming, context management, and the CLI."""

from dataclasses import asdict
import os
from pathlib import Path
import pickle
import string
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from chat import Tokenizer, choose_checkpoint, discover_checkpoints, generate_tokens, load_checkpoint, prepare_prompt, stream_text, terminal_text
from model import GPT, GPTConfig


ROOT = Path(__file__).resolve().parents[1]


def character_metadata():
    characters = '~' + string.ascii_letters + ' :\n'
    return {'stoi': {char: i for i, char in enumerate(characters)},
            'itos': dict(enumerate(characters))}


class ChatTests(unittest.TestCase):
    def test_newest_checkpoint_is_default_and_explicit_choices_override_it(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            older = root / 'runs' / 'alphabetically-last' / 'ckpt.pt'
            newest = root / 'runs' / 'alphabetically-first' / 'ckpt.pt'
            for path, timestamp in ((older, 100), (newest, 200)):
                path.parent.mkdir(parents=True)
                path.write_bytes(b'checkpoint')
                os.utime(path, (timestamp, timestamp))
            # A newer directory without a checkpoint must not become the default.
            (root / 'runs' / 'unfinished').mkdir()
            args = SimpleNamespace(checkpoint=None, out_dir=None, select=False)
            with patch('chat.ROOT', root), patch('builtins.input', side_effect=AssertionError('Unexpected prompt')):
                self.assertEqual(discover_checkpoints(), [newest, older])
                self.assertEqual(choose_checkpoint(args), newest)
                args.out_dir = str(older.parent)
                self.assertEqual(choose_checkpoint(args), older)
                args.out_dir = None
                args.checkpoint = str(older)
                self.assertEqual(choose_checkpoint(args), older)
            args.checkpoint = None
            args.select = True
            with patch('chat.ROOT', root), patch('builtins.input', return_value='2'):
                self.assertEqual(choose_checkpoint(args), older)

    def test_default_without_saved_runs_explains_how_to_select_a_checkpoint(self):
        with TemporaryDirectory() as directory, patch('chat.ROOT', Path(directory)):
            args = SimpleNamespace(checkpoint=None, out_dir=None, select=False)
            with self.assertRaisesRegex(ValueError, 'No checkpoints found in runs/'):
                choose_checkpoint(args)

    def test_character_checkpoint_and_terminal_commands(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = character_metadata()
            with (root / 'meta.pkl').open('wb') as destination:
                pickle.dump(metadata, destination)
            config = GPTConfig(n_layer=1, n_head=2, n_embd=8, block_size=64,
                               vocab_size=len(metadata['stoi']) + 2)
            model = GPT(config)
            # Zero weights make greedy output deterministic: tokenizer ID zero, '~'.
            state = {'_orig_mod.' + key: torch.zeros_like(value)
                     for key, value in model.state_dict().items()}
            checkpoint = root / 'ckpt.pt'
            torch.save({'model_args': asdict(config), 'model': state, 'iter_num': 7,
                        'config': {'dataset': str(root)}}, checkpoint)
            loaded, tokenizer, step = load_checkpoint(checkpoint, torch.device('cpu'))
            self.assertFalse(loaded.training)
            self.assertEqual(step, 7)
            self.assertEqual(loaded.get_num_params(False), model.get_num_params(False))
            self.assertEqual(tokenizer.encode('~'), [0])
            self.assertTrue(all(torch.count_nonzero(param) == 0 for param in loaded.parameters()))

            result = subprocess.run(
                [sys.executable, 'chat.py', '--checkpoint', str(checkpoint), '--device=cpu',
                 '--max_new_tokens=4', '--temperature=0'],
                input='hello\nworld\n/reset\n/help\n/exit\n', cwd=ROOT,
                env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.count('Model> ~~~~'), 2, result.stdout)
            self.assertIn('Conversation cleared.', result.stdout)
            self.assertIn('Goodbye.', result.stdout)

    def test_context_trims_complete_turns_and_raw_omits_labels(self):
        tokenizer = Tokenizer(character_metadata(), 64)
        history = [('old', 'first'), ('recent', 'second')]
        expected = 'User: recent\nAssistant: second\nUser: hello\nAssistant:'
        tokens, kept, trimmed = prepare_prompt(history, 'hello', tokenizer, len(expected))
        self.assertEqual(tokens, tokenizer.encode(expected))
        self.assertEqual(kept, history[-1:])
        self.assertTrue(trimmed)
        self.assertEqual(len(history), 2)
        tokens, kept, trimmed = prepare_prompt(history, 'hello', tokenizer, 5, raw=True)
        self.assertEqual(tokens, tokenizer.encode('hello'))
        self.assertEqual(kept, [])
        self.assertFalse(trimmed)
        with self.assertRaisesRegex(ValueError, 'Shorten it'):
            prepare_prompt(history, 'hello', tokenizer, 5)

    def test_character_tokenizer_rejects_unknown_characters_and_missing_metadata(self):
        tokenizer = Tokenizer(character_metadata(), 64)
        with self.assertRaisesRegex(ValueError, 'not in this model'):
            tokenizer.encode('🙂')
        with self.assertRaisesRegex(ValueError, 'metadata is missing'):
            Tokenizer({}, 64)

    def test_streaming_handles_split_utf8_and_role_delimiters(self):
        chunks = [b'caf\xc3', b'\xa9\nUs', b'er: ignored']
        tokenizer = SimpleNamespace(token_bytes=lambda index: chunks[index])
        self.assertEqual(''.join(stream_text(range(3), tokenizer)), 'café')
        self.assertEqual(''.join(stream_text(range(3), tokenizer, stops=())), 'café\nUser: ignored')
        chunks[:] = [b'answer\nAss', b'istant: ignored']
        self.assertEqual(''.join(stream_text(range(2), tokenizer)), 'answer')
        chunks[:] = [b'answer\nUs']
        self.assertEqual(''.join(stream_text(range(1), tokenizer)), 'answer\nUs')
        self.assertEqual(terminal_text('a\x1b\x00\x7fb\n\tc'), 'ab\n\tc')

    def test_generation_masks_padding_caps_context_and_stops_on_eos(self):
        lengths = []

        class FixedModel:
            config = SimpleNamespace(block_size=2)

            def __call__(self, tokens):
                lengths.append(tokens.shape[1])
                return torch.tensor([[[0., 10., 0., 100.]]]), None

        tokenizer = SimpleNamespace(vocab_size=3, eos_token=2)
        args = (FixedModel(), [0, 0], tokenizer, torch.device('cpu'), torch.float32, 5)
        self.assertEqual(list(generate_tokens(*args, temperature=0, top_k=200)), [1] * 5)
        self.assertEqual(lengths, [2] * 5)
        self.assertEqual(list(generate_tokens(*args, temperature=1, top_k=1)), [1] * 5)
        tokenizer.eos_token = 1
        self.assertEqual(list(generate_tokens(*args, temperature=0, top_k=200)), [])
        self.assertFalse(torch.is_inference_mode_enabled())


if __name__ == '__main__':
    unittest.main()
