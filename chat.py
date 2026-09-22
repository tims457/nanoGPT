"""Interactive text generation from saved nanoGPT checkpoints."""

import argparse
import codecs
from contextlib import closing, nullcontext
import math
from pathlib import Path
import pickle

import tiktoken
import torch

from model import GPT, GPTConfig


ROOT = Path(__file__).resolve().parent
STOP_STRINGS = ('\nUser:', '\nAssistant:')
HELP = '/reset clears conversation history; /help shows commands; /exit quits.\nCtrl+C stops a reply; Ctrl+D exits.'


class Tokenizer:
    def __init__(self, metadata, model_vocab_size):
        self.encoding = None
        self.bpe = None
        self.eos_token = None
        if metadata.get('tokenizer') == 'byte_bpe':
            from tokenizers import Tokenizer as BPETokenizer
            self.bpe = BPETokenizer.from_str(metadata['tokenizer_json'])
            self.vocab_size = self.bpe.get_vocab_size()
            self.eos_token = metadata['eos_token']
            self.byte_table = metadata['token_bytes']
            self.name = f'byte-level BPE ({self.vocab_size} tokens)'
            if len(self.byte_table) != self.vocab_size:
                raise ValueError('Tokenizer byte table does not match its vocabulary')
        elif 'stoi' in metadata and 'itos' in metadata:
            self.stoi, self.itos = metadata['stoi'], metadata['itos']
            self.vocab_size = len(self.stoi)
            self.name = 'characters'
            if (set(self.itos) != set(range(self.vocab_size))
                    or any(self.itos.get(index) != char for char, index in self.stoi.items())):
                raise ValueError('Character tokenizer IDs must be contiguous and stoi/itos must agree')
        else:
            if metadata.get('tokenizer', 'gpt2') != 'gpt2':
                raise ValueError('Only GPT-2, byte-level BPE, and character tokenizers are supported')
            if model_vocab_size < 50257:
                raise ValueError('Character tokenizer metadata is missing; pass --meta=/path/to/meta.pkl')
            self.encoding = tiktoken.get_encoding('gpt2')
            self.vocab_size = self.encoding.n_vocab
            self.eos_token = self.encoding.eot_token
            self.name = 'GPT-2'
        if self.vocab_size > model_vocab_size:
            raise ValueError('Tokenizer vocabulary is larger than the checkpoint vocabulary')

    def encode(self, text):
        if self.bpe is not None:
            return self.bpe.encode(text, add_special_tokens=False).ids
        if self.encoding is not None:
            return self.encoding.encode_ordinary(text)
        try:
            return [self.stoi[char] for char in text]
        except KeyError as error:
            raise ValueError(f'Character {error.args[0]!r} is not in this model\'s vocabulary') from error

    def token_bytes(self, token):
        if self.bpe is not None:
            return self.byte_table[token]
        if self.encoding is not None:
            return self.encoding.decode_single_token_bytes(token)
        return self.itos[token].encode('utf-8')


def discover_checkpoints():
    paths = (path for path in (ROOT / 'runs').glob('*/ckpt.pt') if path.is_file())
    return sorted(paths, key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)


def choose_checkpoint(args):
    if args.checkpoint:
        return Path(args.checkpoint).expanduser().resolve()
    if args.out_dir:
        return Path(args.out_dir).expanduser().resolve() / 'ckpt.pt'
    paths = discover_checkpoints()
    if not paths:
        raise ValueError('No checkpoints found in runs/; pass --out_dir or --checkpoint')
    if not args.select or len(paths) == 1:
        return paths[0]
    print('Saved models (most recent first):')
    for index, path in enumerate(paths, 1):
        print(f'  {index}. {path.parent.name}')
    while True:
        choice = input('Select model [1], or q to quit: ').strip()
        if choice.lower() in ('q', 'quit', 'exit'):
            return None
        if not choice:
            return paths[0]
        if choice.isdigit() and 1 <= int(choice) <= len(paths):
            return paths[int(choice) - 1]
        print(f'Enter a number from 1 to {len(paths)}.')


def load_checkpoint(path, device, meta_path=None):
    # Keep optimizer tensors on CPU and release them before moving model weights to GPU.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if 'model_args' not in checkpoint or 'model' not in checkpoint:
        raise ValueError('Expected a nanoGPT checkpoint containing model_args and model')
    config = GPTConfig(**checkpoint['model_args'])
    model = GPT(config)
    state = {key.removeprefix('_orig_mod.'): value for key, value in checkpoint['model'].items()}
    model.load_state_dict(state)
    metadata = {}
    if meta_path:
        path_to_meta = Path(meta_path).expanduser()
    else:
        dataset = checkpoint.get('config', {}).get('dataset')
        path_to_meta = ROOT / 'data' / dataset / 'meta.pkl' if dataset else None
        if path_to_meta is not None and not path_to_meta.exists():
            path_to_meta = None
    if path_to_meta is not None:
        with path_to_meta.open('rb') as source:
            metadata = pickle.load(source)
    tokenizer = Tokenizer(metadata, config.vocab_size)
    step = checkpoint.get('iter_num', 'unknown')
    del checkpoint, state
    model.eval().to(device)
    return model, tokenizer, step


def prepare_prompt(history, message, tokenizer, block_size, raw=False):
    """Drop oldest complete turns to fit the prompt; never silently cut user input."""
    kept = [] if raw else list(history)
    while True:
        text = message if raw else ''.join(
            f'User: {user}\nAssistant: {reply}\n' for user, reply in kept
        ) + f'User: {message}\nAssistant:'
        tokens = tokenizer.encode(text)
        if not tokens:
            raise ValueError('Enter a nonempty prompt')
        if len(tokens) <= block_size:
            return tokens, kept, len(kept) < len(history) and not raw
        if not kept:
            raise ValueError(f'Message needs {len(tokens)} tokens; context limit is {block_size}. Shorten it.')
        kept.pop(0)


def generate_tokens(model, prompt, tokenizer, device, dtype, max_new_tokens, temperature, top_k):
    context = nullcontext() if device.type != 'cuda' or dtype == torch.float32 else torch.autocast('cuda', dtype=dtype)
    with torch.inference_mode(), context:
        tokens = torch.tensor([prompt], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            logits, _ = model(tokens[:, -model.config.block_size:])
            logits = logits[:, -1, :].float()
            # nanoGPT can pad its vocabulary beyond the actual tokenizer IDs.
            logits[:, tokenizer.vocab_size:] = -float('inf')
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    cutoff = torch.topk(logits, min(top_k, tokenizer.vocab_size)).values[:, [-1]]
                    logits[logits < cutoff] = -float('inf')
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            token = next_token.item()
            if token == tokenizer.eos_token:
                break
            tokens = torch.cat((tokens, next_token), dim=1)[:, -model.config.block_size:]
            yield token


def stream_text(tokens, tokenizer, stops=STOP_STRINGS):
    """Decode UTF-8 incrementally and withhold incomplete conversation delimiters."""
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    pending = ''
    for token in tokens:
        pending += decoder.decode(tokenizer.token_bytes(token))
        positions = [pending.find(stop) for stop in stops if stop in pending]
        if positions:
            yield pending[:min(positions)]
            return
        held = max((length for stop in stops for length in range(1, len(stop))
                    if pending.endswith(stop[:length])), default=0)
        ready = pending[:-held] if held else pending
        pending = pending[-held:] if held else ''
        if ready:
            yield ready
    yield pending + decoder.decode(b'', final=True)


def terminal_text(text):
    # Model output is text, not terminal control sequences.
    return ''.join(char for char in text if char in '\n\t' or (ord(char) >= 32 and not 127 <= ord(char) <= 159))


def chat_loop(model, tokenizer, args, device, dtype):
    history = []
    print('Text-completion checkpoint: conversational behavior depends on its training.')
    print('Raw completion mode: each prompt is independent.' if args.raw else 'Conversation history is enabled.')
    print(HELP)
    while True:
        try:
            message = input('\nPrompt> ' if args.raw else '\nYou> ')
        except EOFError:
            print('\nGoodbye.')
            return
        except KeyboardInterrupt:
            print('\nUse /exit or Ctrl+D to quit.')
            continue
        command = message.strip()
        if command in ('/exit', '/quit'):
            print('Goodbye.')
            return
        if command == '/reset':
            history.clear()
            print('Conversation cleared.')
            continue
        if command == '/help':
            print(HELP)
            continue
        if not command:
            continue
        if command.startswith('/'):
            print('Unknown command. Use /help for available commands.')
            continue
        try:
            prompt, kept, trimmed = prepare_prompt(history, message, tokenizer, model.config.block_size, args.raw)
        except ValueError as error:
            print(f'Input error: {error}')
            continue
        if trimmed:
            print('(Older turns removed to fit the context window.)')
        reply = ''
        print('Model> ', end='', flush=True)
        try:
            with closing(generate_tokens(model, prompt, tokenizer, device, dtype,
                                         args.max_new_tokens, args.temperature, args.top_k)) as tokens:
                for chunk in stream_text(tokens, tokenizer, () if args.raw else STOP_STRINGS):
                    reply += chunk
                    print(terminal_text(chunk), end='', flush=True)
        except KeyboardInterrupt:
            print(' [interrupted]', end='')
        print()
        if not args.raw:
            history = kept + [(message, reply)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog='By default, load the most recently modified runs/*/ckpt.pt.')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--out_dir', '--out-dir', help='Directory containing ckpt.pt')
    source.add_argument('--checkpoint', help='Path to a saved nanoGPT checkpoint')
    source.add_argument('--select', action='store_true', help='Choose a saved run interactively')
    parser.add_argument('--list-models', action='store_true', help='List checkpoints in runs/, newest first')
    parser.add_argument('--meta', help='Tokenizer meta.pkl override, useful for moved character checkpoints')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0, or mps')
    parser.add_argument('--dtype', choices=('auto', 'float32', 'float16', 'bfloat16'), default='auto')
    parser.add_argument('--max_new_tokens', '--max-new-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.8, help='0 for greedy decoding')
    parser.add_argument('--top_k', '--top-k', type=int, default=200, help='0 disables top-k filtering')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--raw', action='store_true', help='Generate independent text continuations without chat labels')
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.top_k < 0 or not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error('max_new_tokens must be positive; temperature and top_k must be nonnegative')
    if args.list_models:
        for path in discover_checkpoints():
            print(path)
        return
    try:
        path = choose_checkpoint(args)
        if path is None:
            return
        device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable. Use a CUDA PyTorch build with GPU access, or --device=cpu.')
        if args.dtype == 'auto':
            dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == 'cuda' else torch.float32
        else:
            dtype = getattr(torch, args.dtype)
        if device.type != 'cuda' and dtype != torch.float32:
            raise ValueError('Use --dtype=float32 or auto with CPU/MPS')
        torch.manual_seed(args.seed)
        print(f'Loading {path} ...', flush=True)
        model, tokenizer, step = load_checkpoint(path, device, args.meta)
        print(f'{model.get_num_params(non_embedding=False):,} parameters | step {step} | '
              f'context {model.config.block_size} | {tokenizer.name} | {device} / {str(dtype).removeprefix("torch.")}')
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.error(str(error))
    except (EOFError, KeyboardInterrupt):
        print('\nGoodbye.')
        return
    try:
        import readline  # enables line editing/history on supported terminals
    except ImportError:
        pass
    chat_loop(model, tokenizer, args, device, dtype)


if __name__ == '__main__':
    main()
