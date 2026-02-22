# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.num_entities = args.entity_size
        self.num_relations = args.rel_size
        self.dim = args.dim
        self.device = torch.device("cuda" if args.use_gpu else "cpu")
        
        self.decoder = nn.Sequential(
        nn.Linear(3 * self.dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)
        )

        # =========================
        # Embeddings
        # =========================
        self.entity_emb = nn.Embedding(self.num_entities, self.dim)
        self.relation_emb = nn.Embedding(self.num_relations, self.dim)

        # =========================
        # R-GCN parameters
        # =========================
        self.W_rel = nn.Parameter(
            torch.randn(self.num_relations, self.dim, self.dim)
        )
        self.W_self = nn.Parameter(torch.randn(self.dim, self.dim))

        # =========================
        # Build graph from train ∪ aux
        # =========================
        edge_src = []
        edge_dst = []
        edge_type = []

        def load_edges(file_path):
            with open(file_path) as f:
                for line in f:
                    h, r, t = map(int, line.strip().split('\t'))
                    edge_src.append(h)
                    edge_dst.append(t)
                    edge_type.append(r)

                    # add inverse edge
                    edge_src.append(t)
                    edge_dst.append(h)
                    edge_type.append(r)

        load_edges(args.train_file)
        load_edges(args.auxiliary_file)

        self.edge_src = torch.tensor(edge_src, dtype=torch.long)
        self.edge_dst = torch.tensor(edge_dst, dtype=torch.long)
        self.edge_type = torch.tensor(edge_type, dtype=torch.long)

        self.to(self.device)

    # ======================================================
    # R-GCN forward (1 layer)
    # ======================================================
    def rgcn_layer(self):
        x = self.entity_emb.weight  # (N, D)

        # message passing indices
        src = self.edge_src.to(self.device)
        dst = self.edge_dst.to(self.device)
        rel = self.edge_type.to(self.device)

        # self-loop contribution
        out = torch.matmul(x, self.W_self)  # (N, D)

        # relation-specific messages
        x_src = x[src]                      # (E, D)
        W_rel = self.W_rel[rel]             # (E, D, D)

        messages = torch.bmm(
            x_src.unsqueeze(1),             # (E, 1, D)
            W_rel                            # (E, D, D)
        ).squeeze(1)                        # (E, D)

        # aggregate
        out.index_add_(0, dst, messages)

        # degree normalization
        deg = torch.zeros(self.num_entities, device=self.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        deg = deg.clamp(min=1).unsqueeze(1)

        out = out / deg

        return F.relu(out)

    def score(self, h, r, t, entity_repr):
        h_e = entity_repr[h]
        r_e = self.relation_emb(r)
        t_e = entity_repr[t]

        x = torch.cat([h_e, r_e, t_e], dim=1)
        return self.decoder(x).squeeze(1)

    # ======================================================
    # Training step
    # ======================================================
    def train_step(self, positive, negative):
        entity_repr = self.rgcn_layer()
        entity_repr = F.normalize(entity_repr, p=2, dim=1)

        pos_h = torch.tensor([x[0] for x in positive], dtype=torch.long, device=self.device)
        pos_r = torch.tensor([x[1] for x in positive], dtype=torch.long, device=self.device)
        pos_t = torch.tensor([x[2] for x in positive], dtype=torch.long, device=self.device)

        neg_h = torch.tensor([x[0] for x in negative], dtype=torch.long, device=self.device)
        neg_r = torch.tensor([x[1] for x in negative], dtype=torch.long, device=self.device)
        neg_t = torch.tensor([x[2] for x in negative], dtype=torch.long, device=self.device)

        pos_score = self.score(pos_h, pos_r, pos_t, entity_repr)
        neg_score = self.score(neg_h, neg_r, neg_t, entity_repr)

        pos_label = torch.ones_like(pos_score)
        neg_label = torch.zeros_like(neg_score)

        loss_pos = F.binary_cross_entropy_with_logits(pos_score, pos_label)
        loss_neg = F.binary_cross_entropy_with_logits(neg_score, neg_label)

        loss = loss_pos + loss_neg
        return loss

    # ======================================================
    # Evaluation
    # ======================================================
    def get_scores(self, batch):
        with torch.no_grad():
            entity_repr = self.rgcn_layer()
            entity_repr = F.normalize(entity_repr, p=2, dim=1)

            h = torch.tensor([x[0] for x in batch], dtype=torch.long, device=self.device)
            r = torch.tensor([x[1] for x in batch], dtype=torch.long, device=self.device)
            t = torch.tensor([x[2] for x in batch], dtype=torch.long, device=self.device)

            score = self.score(h, r, t, entity_repr)
            return torch.sigmoid(score)