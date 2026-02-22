# rgat_main.py

import sys
import random
import datetime
from collections import defaultdict
from more_itertools import chunked

import torch
import torch.optim as optim

import rgat_model


# ======================================================
# Utility
# ======================================================

def trace(*args):
    print(datetime.datetime.now().strftime('%H:%M:%S') + ' ' + ' '.join(map(str,args)))


# ======================================================
# Global containers
# ======================================================

train_data = []
aux_data = []
dev_data = []
test_data = []

gold_heads = defaultdict(set)
gold_tails = defaultdict(set)

candidate_heads = defaultdict(set)
candidate_tails = defaultdict(set)


# ======================================================
# Load datasets
# ======================================================

def load_dataset(args):

    trace('load train')
    for line in open(args.train_file):
        h, r, t = map(int, line.strip().split('\t'))
        train_data.append((h, r, t))

    trace('load aux')
    for line in open(args.auxiliary_file):
        h, r, t = map(int, line.strip().split('\t'))
        aux_data.append((h, r, t))

    trace('load dev')
    for line in open(args.dev_file):
        h, r, t, l = map(int, line.strip().split('\t'))
        dev_data.append((h, r, t, l))

    trace('load test')
    for line in open(args.test_file):
        h, r, t, l = map(int, line.strip().split('\t'))
        test_data.append((h, r, t, l))


# ======================================================
# Negative sampling (train+aux)
# ======================================================

def init_negative_sampling():

    for (h, r, t) in train_data + aux_data:

        candidate_heads[r].add(h)
        candidate_tails[r].add(t)

        gold_heads[(r, t)].add(h)
        gold_tails[(h, r)].add(t)

    for r in candidate_heads:
        candidate_heads[r] = list(candidate_heads[r])

    for r in candidate_tails:
        candidate_tails[r] = list(candidate_tails[r])


def generate_batch(batch_size):

    random.shuffle(train_data)

    positive, negative = [], []

    for (h, r, t) in train_data:

        if random.random() < 0.5:
            # corrupt head
            cand = random.choice(candidate_heads[r])
            while cand in gold_heads[(r, t)]:
                cand = random.choice(candidate_heads[r])
            h_neg, t_neg = cand, t
        else:
            # corrupt tail
            cand = random.choice(candidate_tails[r])
            while cand in gold_tails[(h, r)]:
                cand = random.choice(candidate_tails[r])
            h_neg, t_neg = h, cand

        positive.append((h, r, t))
        negative.append((h_neg, r, t_neg))

        if len(positive) == batch_size:
            yield positive, negative
            positive, negative = [], []

    if len(positive) > 0:
        yield positive, negative


# ======================================================
# Train
# ======================================================

def train(model, optimizer, args):

    model.train()

    total_loss = 0

    for positive, negative in generate_batch(args.batch_size):

        optimizer.zero_grad()

        loss = model.train_step(positive, negative)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


# ======================================================
# Evaluation
# ======================================================

def evaluate(model, data, args):

    model.eval()

    correct = 0
    total = 0

    for batch in chunked(data, args.test_batch_size):

        triples = [(h, r, t) for (h, r, t, l) in batch]

        device = next(model.parameters()).device

        labels = torch.tensor(
            [l for (_, _, _, l) in batch],
            dtype=torch.float,
            device=device
        )

        probs = model.get_scores(triples)

        preds = (probs >= 0.5).float()

        correct += (preds == labels).sum().item()
        total += len(labels)

    return correct / total

def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Hace operaciones determinísticas (más lento pero estable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ======================================================
# Main
# ======================================================

def main(args):

    load_dataset(args)
    print("train size:", len(train_data))
    print("aux size:", len(aux_data))
    print("dev size:", len(dev_data))
    print("test size:", len(test_data))
    init_negative_sampling()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = rgat_model.Model(args, train_data, aux_data).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-5)

    for epoch in range(args.epochs):

        loss = train(model, optimizer, args)

        dev_acc = evaluate(model, dev_data, args)
        test_acc = evaluate(model, test_data, args)

        trace("epoch:", epoch,
              "loss:", loss,
              "dev:", dev_acc,
              "test:", test_acc)


# ======================================================
# Args
# ======================================================

from argparse import ArgumentParser

def argument():

    p = ArgumentParser()

    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--entity_size', type=int, default=38195)
    p.add_argument('--rel_size', type=int, default=11)

    p.add_argument('--batch_size', type=int, default=5000)
    p.add_argument('--test_batch_size', type=int, default=20000)

    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--lr', type=float, default=0.0005)

    p.add_argument('--target_dir', default='head-1000')

    args = p.parse_args()

    base = 'datasets/' + args.target_dir + '/'
    args.train_file = base + 'train.txt'
    args.dev_file = base + 'dev.txt'
    args.test_file = base + 'test.txt'
    args.auxiliary_file = base + 'aux_file.txt'

    return args


if __name__ == '__main__':
    args = argument()
    print(args)
    set_seed(args.seed)
    main(args)