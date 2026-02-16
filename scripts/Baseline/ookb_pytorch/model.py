import torch
import torch.nn as nn
import torch.nn.functional as F


class OOKB(nn.Module):
    """
    OOKB order=1 propagation + TransE scoring
    + Absolute Margin Objective
    """

    def __init__(self, num_entities, num_relations, emb_dim, known_entities):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.emb_dim = emb_dim

        # Base entity embeddings
        self.entity_emb = nn.Embedding(num_entities, emb_dim)

        # Relation embeddings (TransE)
        self.rel_emb = nn.Embedding(num_relations, emb_dim)

        # Relation transformation matrices (OOKB propagation)
        self.rel_weight = nn.Parameter(
            torch.Tensor(2 * num_relations, emb_dim, emb_dim)
        )

        # Mask for known entities
        mask = torch.zeros(num_entities)
        for e in known_entities:
            mask[e] = 1.0
        self.register_buffer("known_mask", mask)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        nn.init.xavier_uniform_(self.rel_weight)

    def forward(self, edge_index, edge_type):
        """
        Order=1 OOKB operator:
        Known entities keep base embedding.
        OOKB entities use aggregated neighbors.
        """

        x0 = self.entity_emb.weight
        x0 = x0 * self.known_mask.unsqueeze(-1)

        row, col = edge_index
        out = torch.zeros_like(x0)

        for r in range(2 * self.num_relations):
            mask = (edge_type == r)
            if mask.sum() == 0:
                continue

            src = row[mask]
            dst = col[mask]
            W_r = self.rel_weight[r]

            messages = torch.matmul(x0[src], W_r.T)
            out.index_add_(0, dst, messages)

        deg = torch.zeros(self.num_entities, device=x0.device)
        deg.index_add_(0, col, torch.ones_like(col, dtype=torch.float))
        deg = deg.clamp(min=1).unsqueeze(-1)

        H1 = out / deg

        mask = self.known_mask.unsqueeze(-1)
        H_final = mask * x0 + (1 - mask) * H1
        H_final = F.normalize(H_final, p=2, dim=1)

        return H_final

    def score(self, h, r, t, entity_repr):
        """
        TransE scoring:
        f(h,r,t) = -|| h + r - t ||_1
        """
        h_e = entity_repr[h]
        t_e = entity_repr[t]
        r_e = self.rel_emb(r)

        return torch.norm(h_e + r_e - t_e, p=1, dim=-1)

    def normalize_embeddings(self):
        with torch.no_grad():
            self.entity_emb.weight.data = F.normalize(
                self.entity_emb.weight.data, p=2, dim=1
            )
            self.rel_emb.weight.data = F.normalize(
                self.rel_emb.weight.data, p=2, dim=1
            )

def absolute_margin_loss(pos_score, neg_score, gamma=1.0):
    pos_loss = torch.abs(pos_score - 0.0)
    neg_loss = torch.abs(neg_score - gamma)
    return torch.mean(pos_loss + neg_loss)
