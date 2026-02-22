# -*- coding: utf-8 -*-

import os
import sys
import random
import datetime
from collections import defaultdict
from more_itertools import chunked

import torch
import torch.optim as optim

import rgcn_model


# ======================================================
# Utility
# ======================================================
def trace(*args):
    print(datetime.datetime.now().strftime('%H:%M:%S') + ' ' + ' '.join(map(str, args)))


# ======================================================
# Global containers (igual que original)
# ======================================================
train_data = []
dev_data = []
test_data = []

gold_heads = defaultdict(set)
gold_tails = defaultdict(set)

tail_per_head = defaultdict(set)
head_per_tail = defaultdict(set)

candidate_heads = defaultdict(set)
candidate_tails = defaultdict(set)

trfreq = defaultdict(int)


# ======================================================
# Dataset properties (para negative sampling)
# ======================================================
def init_property_of_dataset(args):
    global gold_heads, gold_tails
    global candidate_heads, candidate_tails
    global tail_per_head, head_per_tail

    trace('load train properties')

    for line in open(args.train_file):
        h, r, t = map(int, line.strip().split('\t'))

        candidate_heads[r].add(h)
        candidate_tails[r].add(t)

        gold_heads[(r, t)].add(h)
        gold_tails[(h, r)].add(t)

        tail_per_head[h].add(t)
        head_per_tail[t].add(h)

    for r in candidate_heads:
        candidate_heads[r] = list(candidate_heads[r])
    for r in candidate_tails:
        candidate_tails[r] = list(candidate_tails[r])

    for h in tail_per_head:
        tail_per_head[h] = len(tail_per_head[h]) + 0.0
    for t in head_per_tail:
        head_per_tail[t] = len(head_per_tail[t]) + 0.0


# ======================================================
# Load datasets
# ======================================================
def load_dataset(args):
    global train_data, dev_data, test_data, trfreq

    trace('load train')
    for line in open(args.train_file):
        h, r, t = map(int, line.strip().split('\t'))
        train_data.append((h, r, t))
        trfreq[r] += 1

    for r in trfreq:
        trfreq[r] = args.train_size / (float(trfreq[r]) * len(trfreq))

    trace('load dev')
    for line in open(args.dev_file):
        h, r, t, l = map(int, line.strip().split('\t'))
        dev_data.append((h, r, t, l))
    trace('dev size:', len(dev_data))

    trace('load test')
    for line in open(args.test_file):
        h, r, t, l = map(int, line.strip().split('\t'))
        test_data.append((h, r, t, l))
    trace('test size:', len(test_data))


# ======================================================
# Negative sampling generator
# ======================================================
def generator_train_with_corruption(args):
    skip_rate = args.train_size / float(len(train_data))

    positive, negative = [], []
    random.shuffle(train_data)

    for i in range(len(train_data)):
        h, r, t = train_data[i]

        if args.is_balanced_tr:
            if random.random() > trfreq[r]:
                continue
        else:
            if random.random() > skip_rate:
                continue

        head_ratio = 0.5
        if args.is_bernoulli_trick:
            head_ratio = tail_per_head[h] / (tail_per_head[h] + head_per_tail[t])

        if random.random() > head_ratio:
            cand = random.choice(candidate_heads[r])
            while cand in gold_heads[(r, t)]:
                cand = random.choice(candidate_heads[r])
            h_neg = cand
            t_neg = t
        else:
            cand = random.choice(candidate_tails[r])
            while cand in gold_tails[(h, r)]:
                cand = random.choice(candidate_tails[r])
            h_neg = h
            t_neg = cand

        if len(positive) == 0 or len(positive) <= args.batch_size:
            positive.append(train_data[i])
            negative.append((h_neg, r, t_neg))
        else:
            yield positive, negative
            positive, negative = [train_data[i]], [(h_neg, r, t_neg)]

    if len(positive) != 0:
        yield positive, negative


# ======================================================
# Train
# ======================================================
def train(args, m, optimizer):
    total_loss = 0
    total_n = 0

    m.train()

    for positive, negative in generator_train_with_corruption(args):
        optimizer.zero_grad()
        loss = m.train_step(positive, negative)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_n += len(positive)

    return total_loss, total_n


# ======================================================
# Evaluation
# ======================================================
def evaluate(args, m, mode='dev'):
    m.eval()

    current_data = dev_data if mode == 'dev' else test_data

    scores = []
    accuracy = []

    for batch in chunked(current_data, args.test_batch_size):
        batch_triples = [(h, r, t) for (h, r, t, l) in batch]
        labels = [l for (_, _, _, l) in batch]

        preds = m.get_scores(batch_triples).cpu().numpy()

        for pred, label in zip(preds, labels):
            if pred >= 0.5:
                accuracy.append(1.0 if label == 1 else 0.0)
            else:
                accuracy.append(0.0 if label == 1 else 1.0)

    acc = sum(accuracy) / len(accuracy)
    trace('\t', mode, acc)
    return acc


# ======================================================
# Main
# ======================================================
def main(args):
    init_property_of_dataset(args)
    load_dataset(args)

    print('relation size:', args.rel_size, 'entity size:', args.entity_size)

    m = rgcn_model.Model(args)

    optimizer = optim.Adam(m.parameters(), lr=args.beta0,weight_decay=1e-5)

    for epoch in range(args.epoch_size):
        lr = args.beta0 / (1.0 + args.beta1 * epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        tr_loss, n_tr = train(args, m, optimizer)
        trace('epoch:', epoch, 'tr Loss:', tr_loss, n_tr)

        evaluate(args, m, 'dev')
        evaluate(args, m, 'test')


# ======================================================
# Arguments
# ======================================================
from argparse import ArgumentParser

def argument():
    p = ArgumentParser()

    p.add_argument('--use_gpu', '-g', default=True, action='store_true')

    p.add_argument('--target_dir', '-tD', default='head-1000')

    p.add_argument('--rel_size', '-Rs', default=11, type=int)
    p.add_argument('--entity_size', '-Es', default=38195, type=int)

    p.add_argument('--dim', '-D', default=100, type=int)

    p.add_argument('--is_balanced_tr', '-iBtr', default=False, action='store_true')
    p.add_argument('--is_bernoulli_trick', '-iBeT', default=True, action='store_false')

    p.add_argument('--train_size', '-trS', default=10000, type=int)
    p.add_argument('--batch_size', '-bS', default=5000, type=int)
    p.add_argument('--test_batch_size', '-tbS', default=20000, type=int)
    p.add_argument('--epoch_size', '-eS', default=10, type=int)

    p.add_argument('--beta0', '-b0', default=0.0005, type=float)
    p.add_argument('--beta1', '-b1', default=0.00001, type=float)

    p = p.parse_args()

    base = 'datasets/' + p.target_dir + '/'
    p.train_file = base + 'train.txt'
    p.dev_file = base + 'dev.txt'
    p.test_file = base + 'test.txt'
    p.auxiliary_file = base + 'aux_file.txt'

    return p


# ======================================================
if __name__ == '__main__':
    args = argument()
    print(args)
    print(' '.join(sys.argv))
    main(args)