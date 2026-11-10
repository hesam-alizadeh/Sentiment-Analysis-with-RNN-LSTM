from pathlib import Path
from collections import Counter
from functools import partial
import re
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", ' ', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return clean_text(text).split()


class Vocab:
    def __init__(self, min_freq: int = 2, max_size: Optional[int] = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word = [PAD_TOKEN, UNK_TOKEN]
        self.counter = Counter()

    def add_tokens(self, tokens: List[str]):
        self.counter.update(tokens)

    def build(self):
        words = [w for w, c in self.counter.items() if c >= self.min_freq]
        words.sort(key=lambda w: (-self.counter[w], w))
        if self.max_size is not None:
            words = words[: max(0, self.max_size - len(self.idx2word))]
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)

    def pad_idx(self):
        return self.word2idx[PAD_TOKEN]

    def unk_idx(self):
        return self.word2idx[UNK_TOKEN]

    def __len__(self):
        return len(self.idx2word)

    def encode(self, tokens: List[str], max_len: Optional[int] = None) -> torch.LongTensor:
        ids = [self.word2idx.get(tok, self.unk_idx()) for tok in tokens]
        if max_len is not None:
            ids = ids[:max_len]
        return torch.tensor(ids, dtype=torch.long)


class IMDBDataset(Dataset):
    def __init__(self, root: str, split: str, vocab: Vocab, max_len: Optional[int] = 200):
        self.root = Path(root)
        self.split = split
        self.vocab = vocab
        self.max_len = max_len
        self.samples: List[Tuple[str, int]] = []

        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f'missing folder: {split_dir}')

        for label_name, label in [('neg', 0), ('pos', 1)]:
            folder = split_dir / label_name
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if path.is_file():
                    self.samples.append((self._read_text(path), label))

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            return path.read_text(encoding='latin-1')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        text, label = self.samples[idx]
        tokens = tokenize(text)
        ids = self.vocab.encode(tokens, max_len=self.max_len)
        return ids, len(ids), torch.tensor(label, dtype=torch.long)


def collate_batch(batch, pad_idx: int):
    texts, lengths, labels = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.stack(labels)

    max_len = int(lengths.max().item()) if len(lengths) else 0
    padded = torch.full((len(texts), max_len), pad_idx, dtype=torch.long)
    for i, ids in enumerate(texts):
        padded[i, : ids.size(0)] = ids

    return padded, lengths, labels


def build_vocab(train_root: str, min_freq: int = 2, max_size: Optional[int] = None) -> Vocab:
    vocab = Vocab(min_freq=min_freq, max_size=max_size)
    root = Path(train_root)
    for label_name in ['neg', 'pos']:
        folder = root / label_name
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                text = path.read_text(encoding='latin-1')
            vocab.add_tokens(tokenize(text))
    vocab.build()
    return vocab


def get_loaders(root: str = 'aclImdb', batch_size: int = 64, max_len: int = 200, min_freq: int = 2):
    root_path = Path(root)
    train_root = root_path / 'train'
    test_root = root_path / 'test'

    vocab = build_vocab(str(train_root), min_freq=min_freq)
    train_ds = IMDBDataset(str(root_path), 'train', vocab=vocab, max_len=max_len)
    test_ds = IMDBDataset(str(root_path), 'test', vocab=vocab, max_len=max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_batch, pad_idx=vocab.pad_idx()),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_batch, pad_idx=vocab.pad_idx()),
    )
    return train_loader, test_loader, vocab


if __name__ == '__main__':
    if Path('aclImdb').is_dir():
        train_loader, test_loader, vocab = get_loaders()
        print('train batches:', len(train_loader))
        print('test batches:', len(test_loader))
        print('vocab size:', len(vocab))
    else:
        print('aclImdb not found in current folder')
