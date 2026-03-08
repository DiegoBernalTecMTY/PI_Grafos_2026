import sys; sys.path.insert(0, '.')
import torch
from train_ikge import IKGENetwork
torch.manual_seed(42)
device = torch.device('cpu')
emb = torch.randn(1000, 300)
model = IKGENetwork(emb, 300, 256, 128, 50, 3, device, dropout=0.0)
model.eval()
B = 4
batch = {
    'head_desc': torch.randint(1, 1000, (B, 50)),
    'tail_desc':  torch.randint(1, 1000, (B, 50)),
    'rel_name':   torch.randint(1, 1000, (B, 10)),
    'head_type':  torch.rand(B, 50), 'tail_type': torch.rand(B, 50),
    'rel_domain': torch.rand(B, 50), 'rel_range':  torch.rand(B, 50),
    'head_len':   torch.tensor([10,15,8,20]),
    'tail_len':   torch.tensor([12,10,20,5]),
}
feat = model.extract_fact_features(batch)
print('feat std across batch:', feat.std(dim=0).mean().item())
print('feat sample:', feat[0,:4].tolist(), '...')
edge_index = torch.zeros(2, 0, dtype=torch.long)
agg = model.aggregator(feat, edge_index)
scores = model(agg)
print('scores:', [round(s,4) for s in scores.tolist()])
print('std:', round(scores.std().item(), 6))
result = 'PASS' if scores.std().item() > 1e-4 else 'FAIL: still collapsed!'
print(result)
# Also test WITH edges to ensure aggregation path is correct
src = torch.tensor([0,1,2,3,0,2])
dst = torch.tensor([1,0,3,2,2,0])
edge_index2 = torch.stack([src, dst])
agg2 = model.aggregator(feat, edge_index2)
scores2 = model(agg2)
print('with-edges scores:', [round(s,4) for s in scores2.tolist()])
print('with-edges std:', round(scores2.std().item(), 6))
print('with-edges:', 'PASS' if scores2.std().item() > 1e-4 else 'FAIL')
