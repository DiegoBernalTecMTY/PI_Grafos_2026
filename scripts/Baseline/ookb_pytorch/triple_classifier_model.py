import torch
import torch.nn as nn
import torch.nn.functional as F


class TripleClassifierModel(nn.Module):

    def __init__(self, num_entities, num_relations, emb_dim, device="cpu"):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.emb_dim = emb_dim
        self.device = device

        # Embeddings
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.rel_emb = nn.Embedding(num_relations, emb_dim)

        # Clasificador MLP simple
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1)
        )

        self.reset_parameters()

    # ------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------

    def forward(self, h, r, t):

        h_e = self.entity_emb(h)
        r_e = self.rel_emb(r)
        t_e = self.entity_emb(t)

        x = torch.cat([h_e, r_e, t_e], dim=1)

        logits = self.classifier(x).squeeze(-1)

        return logits

    # ------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------

    def train_step(self, positive, negative):

        triples = positive + negative
        labels = [1.0] * len(positive) + [0.0] * len(negative)

        h = torch.tensor([h for h, r, t in triples], device=self.device)
        r = torch.tensor([r for h, r, t in triples], device=self.device)
        t = torch.tensor([t for h, r, t in triples], device=self.device)

        labels = torch.tensor(labels, device=self.device)

        logits = self.forward(h, r, t)

        loss = F.binary_cross_entropy_with_logits(logits, labels)

        return loss

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    def get_scores(self, triples):

        with torch.no_grad():

            h = torch.tensor([h for h, r, t, l in triples], device=self.device)
            r = torch.tensor([r for h, r, t, l in triples], device=self.device)
            t = torch.tensor([t for h, r, t, l in triples], device=self.device)

            logits = self.forward(h, r, t)

            probs = torch.sigmoid(logits)

        return probs.cpu().tolist()