# rgat_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================
# Relational Graph Attention Layer (1-head)
# ======================================================

class RelationalAttentionLayer(nn.Module):
    def __init__(self, dim, num_relations, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.num_relations = num_relations

        self.W_r = nn.Parameter(torch.Tensor(num_relations, dim, dim))
        self.att_r = nn.Parameter(torch.Tensor(num_relations, 2 * dim))

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_r)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, edge_type):

        device = x.device
        out = torch.zeros_like(x)

        src = edge_index[0]
        dst = edge_index[1]

        for r in range(self.num_relations):

            mask = (edge_type == r)
            if mask.sum() == 0:
                continue

            src_r = src[mask]
            dst_r = dst[mask]

            x_src = x[src_r]
            x_dst = x[dst_r]

            Wh_src = torch.matmul(x_src, self.W_r[r])
            Wh_dst = torch.matmul(x_dst, self.W_r[r])

            a_input = torch.cat([Wh_src, Wh_dst], dim=1)

            e = F.leaky_relu(
                torch.sum(a_input * self.att_r[r], dim=1)
            )

            # --------- Softmax por destino (vectorizado) ---------

            # restar max por destino para estabilidad
            max_per_dst = torch.zeros(
                x.size(0), device=device
            ).index_reduce_(
                0, dst_r, e, reduce="amax"
            )

            e_exp = torch.exp(e - max_per_dst[dst_r])

            sum_per_dst = torch.zeros(
                x.size(0), device=device
            ).index_add_(
                0, dst_r, e_exp
            )

            alpha = e_exp / (sum_per_dst[dst_r] + 1e-9)

            alpha = self.dropout(alpha)

            messages = Wh_src * alpha.unsqueeze(-1)

            out.index_add_(0, dst_r, messages)

        return out


# ======================================================
# RGAT Model
# ======================================================

class Model(nn.Module):
    def __init__(self, args, train_triples, aux_triples):
        super().__init__()

        self.dim = args.dim
        self.num_entities = args.entity_size
        self.num_relations = args.rel_size

        self.entity_embedding = nn.Embedding(
            self.num_entities, self.dim
        )

        self.relation_embedding = nn.Embedding(
            self.num_relations, self.dim
        )
        
        self.rgat1 = RelationalAttentionLayer(self.dim, self.num_relations, dropout=0.33)
        self.rgat2 = RelationalAttentionLayer(self.dim, self.num_relations, dropout=0.33)

        self.build_graph(train_triples, aux_triples)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    # --------------------------------------------------
    # Grafo estructural = train + aux
    # --------------------------------------------------
    def build_graph(self, train_triples, aux_triples):

        all_triples = train_triples + aux_triples

        heads = [h for (h, r, t) in all_triples]
        rels  = [r for (h, r, t) in all_triples]
        tails = [t for (h, r, t) in all_triples]

        edge_index = torch.tensor(
            [heads, tails],
            dtype=torch.long
        )

        edge_type = torch.tensor(
            rels,
            dtype=torch.long
        )

        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)

    # --------------------------------------------------
    # Embeddings estructurales
    # --------------------------------------------------
    def compute_embeddings(self):

        x0 = self.entity_embedding.weight

        H1 = self.rgat1(x0, self.edge_index, self.edge_type)
        H1 = F.relu(H1)
        H1 = x0 + H1   # residual intermedio

        H2 = self.rgat2(H1, self.edge_index, self.edge_type)

        H = x0 + H2

        return H

    # --------------------------------------------------
    # DistMult score (logits)
    # --------------------------------------------------
    def score(self, H, triples):

        triples = torch.as_tensor(triples, dtype=torch.long, device=H.device)

        h = H[triples[:, 0]]
        r = self.relation_embedding(triples[:, 1])
        t = H[triples[:, 2]]

        logits = torch.sum(h * r * t, dim=1)

        return logits

    # --------------------------------------------------
    # Train step
    # --------------------------------------------------
    def train_step(self, positive, negative):

        device = self.entity_embedding.weight.device

        H = self.compute_embeddings()

        pos_logits = self.score(H, positive)
        neg_logits = self.score(H, negative)

        logits = torch.cat([pos_logits, neg_logits], dim=0)

        device = pos_logits.device

        labels = torch.cat([
            torch.ones(len(pos_logits), device=device),
            torch.zeros(len(neg_logits), device=device)
        ])

        loss = self.loss_fn(logits, labels)

        return loss

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    def get_scores(self, triples):

        H = self.compute_embeddings()
        logits = self.score(H, triples)

        probs = torch.sigmoid(logits)

        return probs