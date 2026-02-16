# -*- coding: utf-8 -*-

import os
import sys
import random
import datetime
from collections import defaultdict
from argparse import ArgumentParser

import torch
import torch.optim as optim

from model import OOKB, absolute_margin_loss


# ------------------------------------------------------------------

def trace(*args):
    print(datetime.datetime.now().strftime('%H:%M:%S') + ' ' + ' '.join(map(str, args)))


# ------------------------------------------------------------------

train_data = []
dev_data = []
test_data = []

gold_heads = defaultdict(set)
gold_tails = defaultdict(set)
gold_relations = dict()

tail_per_head = defaultdict(set)
head_per_tail = defaultdict(set)

candidate_heads = defaultdict(set)
candidate_tails = defaultdict(set)

train_link = defaultdict(set)
aux_link = defaultdict(set)
trfreq = defaultdict(int)


# ------------------------------------------------------------------

def init_property_of_dataset(args):
    global gold_heads, gold_tails, gold_relations
    global candidate_heads, candidate_tails
    global train_link, aux_link

    trace('load train properties')

    for line in open(args.train_file):
        h, r, t = map(int, line.strip().split('\t'))

        candidate_heads[r].add(h)
        candidate_tails[r].add(t)

        gold_heads[(r, t)].add(h)
        gold_tails[(h, r)].add(t)

        tail_per_head[h].add(t)
        head_per_tail[t].add(h)

        train_link[t].add(h)
        train_link[h].add(t)

        gold_relations[(h, t)] = r

    for e in train_link:
        train_link[e] = list(train_link[e])

    for r in candidate_heads:
        candidate_heads[r] = list(candidate_heads[r])
    for r in candidate_tails:
        candidate_tails[r] = list(candidate_tails[r])

    for h in tail_per_head:
        tail_per_head[h] = len(tail_per_head[h]) + 0.0
    for t in head_per_tail:
        head_per_tail[t] = len(head_per_tail[t]) + 0.0

    trace('load auxiliary (OOKB edges)')

    aux_link = defaultdict(set)

    for line in open(args.auxiliary_file):
        h, r, t = map(int, line.strip().split('\t'))

        gold_relations[(h, t)] = r
        aux_link[t].add(h)
        aux_link[h].add(t)

    for e in aux_link:
        aux_link[e] = list(aux_link[e])


# ------------------------------------------------------------------

def parse_line(line):
    return list(map(int, line.strip().split('\t')))


def load_dataset(args):
    global train_data, dev_data, test_data, trfreq

    trace('load train')
    for line in open(args.train_file):
        h, r, t = parse_line(line)
        train_data.append((h, r, t))
        trfreq[r] += 1

    for r in trfreq:
        trfreq[r] = args.train_size / (float(trfreq[r]) * len(trfreq))

    trace('load dev')
    for line in open(args.dev_file):
        h, r, t, l = parse_line(line)
        dev_data.append((h, r, t, l))
    trace('dev size:', len(dev_data))

    trace('load test')
    for line in open(args.test_file):
        h, r, t, l = parse_line(line)
        test_data.append((h, r, t, l))
    trace('test size:', len(test_data))


# ------------------------------------------------------------------

def build_edge_index(args, device):
    """
    Construye edge_index y edge_type incluyendo:
    - train edges
    - auxiliary edges
    - direcciones forward/backward
    """

    edges = []
    edge_types = []

    for line in open(args.train_file):
        h, r, t = map(int, line.strip().split('\t'))

        edges.append((h, t))
        edge_types.append(r)

        edges.append((t, h))
        edge_types.append(r + args.rel_size)

    for line in open(args.auxiliary_file):
        h, r, t = map(int, line.strip().split('\t'))

        edges.append((h, t))
        edge_types.append(r)

        edges.append((t, h))
        edge_types.append(r + args.rel_size)

    edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)
    edge_type = torch.tensor(edge_types, dtype=torch.long).to(device)

    return edge_index, edge_type


# ------------------------------------------------------------------

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
            h = cand
        else:
            cand = random.choice(candidate_tails[r])
            while cand in gold_tails[(h, r)]:
                cand = random.choice(candidate_tails[r])
            t = cand

        if len(positive) <= args.batch_size:
            positive.append(train_data[i])
            negative.append((h, r, t))
        else:
            yield positive, negative
            positive, negative = [train_data[i]], [(h, r, t)]

    if len(positive) != 0:
        yield positive, negative


# ------------------------------------------------------------------

def train(args, model, optimizer, edge_index, edge_type, device):
    Loss = []
    N = 0

    for positive, negative in generator_train_with_corruption(args):

        optimizer.zero_grad()

        entity_repr = model(edge_index, edge_type)

        pos_h = torch.tensor([x[0] for x in positive], device=device)
        pos_r = torch.tensor([x[1] for x in positive], device=device)
        pos_t = torch.tensor([x[2] for x in positive], device=device)

        neg_h = torch.tensor([x[0] for x in negative], device=device)
        neg_r = torch.tensor([x[1] for x in negative], device=device)
        neg_t = torch.tensor([x[2] for x in negative], device=device)

        pos_score = model.score(pos_h, pos_r, pos_t, entity_repr)
        neg_score = model.score(neg_h, neg_r, neg_t, entity_repr)

        loss = absolute_margin_loss(pos_score, neg_score)

        loss.backward()
        optimizer.step()

        model.normalize_embeddings()

        Loss.append(loss.item())
        N += len(positive)

    return sum(Loss), N

def find_best_threshold(scores, labels):
    thresholds = torch.linspace(scores.min(), scores.max(), 200)

    best_acc = 0
    best_t = 0

    for t in thresholds:
        preds = (scores < t).float()
        acc = (preds == labels).float().mean()
        if acc > best_acc:
            best_acc = acc
            best_t = t

    return best_t.item(), best_acc.item()


# ------------------------------------------------------------------

def evaluate(args, model, data, edge_index, edge_type, device, threshold=None):

    model.eval()
    scores_list = []
    labels_list = []

    with torch.no_grad():
        entity_repr = model(edge_index, edge_type)

        for h, r, t, l in data:
            h = torch.tensor([h], device=device)
            r = torch.tensor([r], device=device)
            t = torch.tensor([t], device=device)

            score = model.score(h, r, t, entity_repr).item()

            scores_list.append(score)
            labels_list.append(l)

    scores = torch.tensor(scores_list)
    labels = torch.tensor(labels_list).float()

    if threshold is None:
        threshold, acc = find_best_threshold(scores, labels)
        return threshold, acc
    else:
        preds = (scores < threshold).float()
        acc = (preds == labels).float().mean().item()
        return acc



# ------------------------------------------------------------------

def main(args):

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        f'cuda:{args.gpu_device}' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    )

    init_property_of_dataset(args)
    load_dataset(args)

    edge_index, edge_type = build_edge_index(args, device)

    known_entities = set()
    for h, _, t in train_data:
        known_entities.add(h)
        known_entities.add(t)
    print("Known count:", len(known_entities))


    model = OOKB(
        num_entities=args.entity_size,
        num_relations=args.rel_size,
        emb_dim=args.dim,
        known_entities=known_entities
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.beta0)

    for epoch in range(args.epoch_size):

        lr = args.beta0 / (1.0 + args.beta1 * epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        trLoss, Ntr = train(args, model, optimizer, edge_index, edge_type, device)

        trace('epoch:', epoch, 'tr Loss:', trLoss, Ntr)

        threshold, dev_acc = evaluate(
            args, model, dev_data, edge_index, edge_type, device
        )

        test_acc = evaluate(
            args, model, test_data, edge_index, edge_type, device, threshold
        )

        trace('epoch:', epoch, 'dev:', dev_acc, 'test:', test_acc)



# ------------------------------------------------------------------

def argument():
    p = ArgumentParser()

    p.add_argument('--use_gpu', '-g', action='store_true')
    p.add_argument('--gpu_device', '-gd', default=0, type=int)

    p.add_argument('--target_dir', '-tD', default='head-1000')

    p.add_argument('--rel_size', '-Rs', default=11, type=int)
    p.add_argument('--entity_size', '-Es', default=38195, type=int)

    p.add_argument('--dim', '-D', default=20, type=int)
    p.add_argument('--threshold', '-T', default=0.0, type=float)

    p.add_argument('--is_balanced_tr', '-iBtr', action='store_true')
    p.add_argument('--is_bernoulli_trick', '-iBeT', default=True, action='store_false')

    p.add_argument('--train_size', '-trS', default=1000, type=int)
    p.add_argument('--batch_size', '-bS', default=5000, type=int)
    p.add_argument('--epoch_size', '-eS', default=50, type=int)

    p.add_argument('--beta0', '-b0', default=0.005, type=float)
    p.add_argument('--beta1', '-b1', default=0.00001, type=float)

    p.add_argument('--seed', '-seed', default=0, type=int)

    p = p.parse_args()

    p.train_file = f'datasets/{p.target_dir}/train.txt'
    p.dev_file = f'datasets/{p.target_dir}/dev.txt'
    p.test_file = f'datasets/{p.target_dir}/test.txt'
    p.auxiliary_file = f'datasets/{p.target_dir}/aux_file.txt'

    return p


# ------------------------------------------------------------------

if __name__ == '__main__':
    args = argument()
    print(args)
    print(' '.join(sys.argv))
    main(args)
