import os
import hashlib
import json
import pickle
import requests
import tiktoken
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
if not os.path.exists(input_file_path):
    response = requests.get(data_url, timeout=30)
    response.raise_for_status()
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(response.text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Match the general loader's metadata so training logs the dataset and vocabulary.
metadata = {
    'dataset': 'tiny_shakespeare',
    'source_url': data_url,
    'source_sha256': hashlib.sha256(data.encode('utf-8')).hexdigest(),
    'tokenizer': 'gpt2',
    'vocab_size': enc.n_vocab,
    'dtype': 'uint16',
    'split_method': 'first 90% of characters for training, remaining 10% for validation',
    'train': {'tokens': len(train_ids)},
    'val': {'tokens': len(val_ids)},
}
with open(os.path.join(os.path.dirname(__file__), 'meta.json'), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
    f.write('\n')
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': enc.n_vocab, 'tokenizer': 'gpt2'}, f)

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
