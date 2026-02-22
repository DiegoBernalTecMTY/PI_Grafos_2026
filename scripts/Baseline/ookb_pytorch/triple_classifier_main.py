# -*- coding: utf-8 -*-

import sys
import random
import datetime
from collections import defaultdict
from argparse import ArgumentParser

import torch
import torch.optim as optim

from triple_classifier_model import TripleClassifierModel


# ------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------

def trace(*args):
    print(datetime.datetime.now().strftime('%H:%M:%S') + ' ' + ' '.join(map(str, args)))


# ------------------------------------------------------------------
# Global dataset containers
# ------------------------------------------------------------------

global train_data, dev_data, test_data, aux_data
global gold_heads, gold_tails
global candidate_heads, candidate_tails
global tail_per_head, head_per_tail
global trfreq

train_data = []
dev_data = []
test_data = []
aux_data = []

gold_heads = defaultdict(set)
gold_tails = defaultdict(set)

candidate_heads = defaultdict(set)
candidate_tails = defaultdict(set)

tail_per_head = defaultdict(set)
head_per_tail = defaultdict(set)

trfreq = defaultdict(int)

# ------------------------------------------------------------------
# Dataset preparation
# ------------------------------------------------------------------

def init_property_of_dataset(args):

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


def parse_line(line):
    return list(map(int, line.strip().split('\t')))


def load_dataset(args):

    trace('load train')
    for line in open(args.train_file):
        h, r, t = parse_line(line)
        train_data.append((h, r, t))
        trfreq[r] += 1

    trace('load dev')
    for line in open(args.dev_file):
        h, r, t, l = parse_line(line)
        dev_data.append((h, r, t, l))

    trace('load test')
    for line in open(args.test_file):
        h, r, t, l = parse_line(line)
        test_data.append((h, r, t, l))

    trace('load aux')
    for line in open(args.auxiliary_file):
        h, r, t = parse_line(line)
        aux_data.append((h, r, t))


# ------------------------------------------------------------------
# Negative sampling (igual que antes)
# ------------------------------------------------------------------

def generator_train_with_corruption(args):

    positive, negative = [], []
    random.shuffle(train_data)

    for h, r, t in train_data:

        head_ratio = 0.5

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

        positive.append((h, r, t))
        negative.append((h_neg, r, t_neg))

        if len(positive) >= args.batch_size:
            yield positive, negative
            positive, negative = [], []

    if len(positive) > 0:
        yield positive, negative


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(args, model, optimizer):

    Loss = []

    for positive, negative in generator_train_with_corruption(args):

        loss = model.train_step(positive, negative)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Loss.append(loss.item())

    return sum(Loss)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(model, data):

    scores = model.get_scores(data)
    labels = torch.tensor([l for _, _, _, l in data]).float()

    scores = torch.tensor(scores)

    preds = (scores > 0.5).float()

    acc = (preds == labels).float().mean().item()

    return acc


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):

    # -----------------------------
    # Reset global containers
    # -----------------------------
    global train_data, dev_data, test_data, aux_data
    global gold_heads, gold_tails
    global candidate_heads, candidate_tails
    global tail_per_head, head_per_tail
    global trfreq

    train_data = []
    dev_data = []
    test_data = []
    aux_data = []

    gold_heads = defaultdict(set)
    gold_tails = defaultdict(set)

    candidate_heads = defaultdict(set)
    candidate_tails = defaultdict(set)

    tail_per_head = defaultdict(set)
    head_per_tail = defaultdict(set)

    trfreq = defaultdict(int)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        f'cuda:{args.gpu_device}' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    )

    init_property_of_dataset(args)
    load_dataset(args)

    trace('relation size:', args.rel_size, 'entity size:', args.entity_size)

    model = TripleClassifierModel(
        num_entities=args.entity_size,
        num_relations=args.rel_size,
        emb_dim=args.dim,
        device=device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.beta0, weight_decay=1e-4)

    best_dev = 0.0
    best_test = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0

    for epoch in range(args.epoch_size):

        loss = train(args, model, optimizer)

        dev_acc = evaluate(model, dev_data)
        test_acc = evaluate(model, test_data)

        trace('epoch:', epoch, 'loss:', loss, 'dev:', dev_acc, 'test:', test_acc)

        if dev_acc > best_dev:
            best_dev = dev_acc
            best_test = test_acc
            best_epoch = epoch
            patience_counter = 0

            #torch.save(model.state_dict(), "best_model.pt")

        else:
            patience_counter += 1

        if patience_counter >= patience:
            trace("Early stopping triggered.")
            break

    # -------------------------------------------------
    # Load best model and evaluate once more
    # -------------------------------------------------

    trace("Loading best model from epoch", best_epoch)
    model.load_state_dict(torch.load("best_model.pt"))

    final_dev = evaluate(model, dev_data)
    final_test = evaluate(model, test_data)

    trace("FINAL RESULTS -> dev:", final_dev, "test:", final_test)
    print(f"SEED {args.seed} -> FINAL TEST: {final_test}")
    
    return final_test


# ------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------

def argument():
    p = ArgumentParser()

    p.add_argument('--use_gpu', '-g', action='store_true')
    p.add_argument('--gpu_device', '-gd', default=0, type=int)

    p.add_argument('--target_dir', '-tD', default='head-1000')

    p.add_argument('--rel_size', '-Rs', default=11, type=int)
    p.add_argument('--entity_size', '-Es', default=38195, type=int)

    p.add_argument('--dim', '-D', default=24, type=int)

    p.add_argument('--batch_size', '-bS', default=5000, type=int)
    p.add_argument('--epoch_size', '-eS', default=10, type=int)

    p.add_argument('--beta0', '-b0', default=0.001, type=float)

    p.add_argument('--seed', '-seed', default=0, type=int)

    p = p.parse_args()

    p.train_file = f'datasets/{p.target_dir}/train.txt'
    p.dev_file = f'datasets/{p.target_dir}/dev.txt'
    p.test_file = f'datasets/{p.target_dir}/test.txt'
    p.auxiliary_file = f'datasets/{p.target_dir}/aux_file.txt'

    return p


#if __name__ == '__main__':
#    args = argument()
#    print(args)
#    print(' '.join(sys.argv))
#    main(args)

if __name__ == "__main__":

    seeds = [0, 1, 2]

    results = []

    for s in seeds:
        args = argument()
        args.seed = s
        print(f"\nRunning seed {s}\n")
        final_test = main(args)
        results.append(final_test)

    print("\n----------------------------------")
    print("Final Results:")
    print("Mean:", sum(results)/len(results))
    print("Std:", (sum((x - sum(results)/len(results))**2 for x in results)/len(results))**0.5)