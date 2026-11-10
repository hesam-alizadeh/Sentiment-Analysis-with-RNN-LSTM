import argparse
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from data_loader import get_loaders


def make_model(model_name, vocab_size, device, padding_idx):
    if model_name == 'rnn':
        from rnn_model import SentimentRNN

        model = SentimentRNN(vocab_size=vocab_size, padding_idx=padding_idx)
    elif model_name == 'lstm':
        from lstm_model import SentimentLSTM

        model = SentimentLSTM(vocab_size=vocab_size, padding_idx=padding_idx)
    else:
        raise ValueError('model should be rnn or lstm')
    return model.to(device)


class Trainer:
    def __init__(self, model, train_loader, test_loader, device, lr=1e-3):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.best_test_acc = 0.0

    def run_epoch(self, loader, train=True):
        if train:
            self.model.train(True)
        else:
            self.model.train(False)

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        bar = tqdm(loader, leave=False)
        for inputs, lengths, labels in bar:
            inputs = inputs.to(self.device)
            lengths = lengths.to(self.device)
            labels = labels.to(self.device)

            if train:
                self.optimizer.zero_grad()

            logits = self.model(inputs, lengths)
            loss = self.loss_fn(logits, labels)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += batch_size

            bar.set_postfix(loss=loss.item(), acc=total_correct / max(1, total_count))

        avg_loss = total_loss / max(1, total_count)
        avg_acc = total_correct / max(1, total_count)
        return avg_loss, avg_acc

    def fit(self, epochs=5):
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss, train_acc = self.run_epoch(self.train_loader, train=True)
            test_loss, test_acc = self.run_epoch(self.test_loader, train=False)

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accs.append(test_acc)

            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc

            print(
                f'epoch {epoch:02d} | '
                f'train loss {train_loss:.4f} acc {train_acc:.4f} | '
                f'test loss {test_loss:.4f} acc {test_acc:.4f} | '
                f'best {self.best_test_acc:.4f} | '
                f'{time.time() - start:.1f}s'
            )


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='aclImdb')
    parser.add_argument('--model', type=str, default='rnn', choices=['rnn', 'lstm'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-len', type=int, default=200)
    parser.add_argument('--min-freq', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, vocab = get_loaders(
        root=args.data,
        batch_size=args.batch_size,
        max_len=args.max_len,
        min_freq=args.min_freq,
    )

    model = make_model(args.model, len(vocab), device=device, padding_idx=vocab.pad_idx())
    trainer = Trainer(model, train_loader, test_loader, device, lr=args.lr)
    trainer.fit(args.epochs)


if __name__ == '__main__':
    cli()
