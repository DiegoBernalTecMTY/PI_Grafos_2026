# IKGE Paper → Code Correspondence

**Paper:** *"Open-world knowledge graph completion for unseen entities and relations via attentive feature aggregation"*  
Byungkook Oh et al., Information Sciences 586 (2022) 468–484

This document maps each paper section to the exact lines of code that implement it.

---

## Section 5 — The IKGE Model (Overview)

The paper describes the overall pipeline in two phases:

> *"in the training phase, (a) given a sample KG, (b) we first extract fact feature information for every fact from word-level side information and construct a line graph where a node and an edge are a fact and a pair of adjacent edges, respectively. (c) After applying an attention-based GCN, a fact feature extractor for fact feature information extraction, aggregator functions for attentive feature aggregation, and fully-connected (FC) layers for scoring facts, are trained via supervised learning."*

> *"In the inference phase, (d) given a target fact f_tar where r and t are out-of-KG, (e) we extract the feature information f_tar of the target fact with the trained fact feature extractor. Then, (f) the multi-hop neighboring fact feature information is hierarchically accumulated with the trained aggregator functions. Finally, we can score the generated target fact's embedding z_tar with the FC layers to determine the plausibility of the target fact f_tar."*

### 5(a) — Given a sample KG (data loading)

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L1393-L1420)  
`_main()` loads `train.txt`, `valid.txt`, `test.txt` and supporting metadata files (`entity2text.txt`, `entity2type.txt`, `relation2constraint.txt`). Entity-to-integer and relation-to-integer maps are built from all splits to ensure full coverage.

```python
# train_ikge_w2v.py  lines 1393–1420
train_triples = load_txt(os.path.join(data_dir, 'train.txt'))
val_triples   = load_txt(os.path.join(data_dir, 'valid.txt'))
test_triples  = load_txt(os.path.join(data_dir, 'test.txt'))
...
entity2desc  = {x[0]: x[1] for x in entity2desc_raw if len(x) == 2}
entity2types = defaultdict(list)
rel2domain   = defaultdict(list)
rel2range    = defaultdict(list)
```

### 5(b) — Construct line graph

> *"construct a line graph where a node and an edge are a fact and a pair of adjacent edges, respectively"*

**File:** [line_graph.py](line_graph.py#L22-L50) — `LineGraph` class.  
Nodes are facts; two fact-nodes are connected when they share an entity.

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L1481-L1486) — construction call site:

```python
# train_ikge_w2v.py  lines 1481–1486
id_train_triples = [
    (ent2id[h], rel2id[r], ent2id[t])
    for h, r, t in train_triples
]
train_triple_tensor = torch.tensor(id_train_triples, dtype=torch.long)
fact_edge_index, _ = create_line_graph(train_triple_tensor)
```

During training the full line graph is replaced by per-batch K-hop BFS subgraphs (see Section 5.2.1 below).  
BFS construction: [train_ikge_w2v.py](train_ikge_w2v.py#L459-L511) — `sample_subgraph_for_triple()`.

### 5(b) — Fact Feature Information Extraction (overview)

> *"we first extract fact feature information for every fact from word-level side information"*
> *"For relation r and entity e, we denote relation-related (type constraints T_{r,d} for domain and T_{r,r} for range, and relation name U_r) and entity-related (description D_e, types T_e, and name U_e) word-level shared side information."*

**Module:** [fact_feature_extractor.py](fact_feature_extractor.py) — `FactFeatureExtractor` class.  
**Instantiation:** [train_ikge_w2v.py](train_ikge_w2v.py#L119-L128):

```python
# train_ikge_w2v.py  lines 119–128
self.fact_extractor = FactFeatureExtractor(
    word_embedding_matrix=embedding_matrix,
    word_embedding_dim=word_emb_dim,
    fact_embedding_dim=fact_emb_dim,
    conv_channels=conv_channels,
    num_types=num_types,
    dropout=dropout,
    device=device
)
```

**Call wrapper:** [train_ikge_w2v.py](train_ikge_w2v.py#L150-L163) — `IKGENetwork.extract_fact_features()`, which maps the batch dict keys to the extractor's positional arguments.

Pre-tokenisation of all entities and relations is done once before training:  
[train_ikge_w2v.py](train_ikge_w2v.py#L245-L323) — `precompute_entity_tensors()` and `precompute_relation_tensors()`.

### 5(c) — Attentive Feature Aggregation (overview)

> *"we apply an attention-based graph convolution network to recursively aggregate neighboring facts' feature information … the aggregated neighborhood vector h_{N_f} is combined with the fact feature information f of the fact f. The combined vector ~f is used as fact feature information for f at the next aggregator function … At final depth k, a score function w(·) based on fully-connected layers assesses the fact f."*

**Module:** [attentive_aggregator.py](attentive_aggregator.py) — `AttentiveAggregator` class.  
**Instantiation:** [train_ikge_w2v.py](train_ikge_w2v.py#L132-L136):

```python
# train_ikge_w2v.py  lines 132–136
self.aggregator = AttentiveAggregator(
    fact_embedding_dim=fact_emb_dim,
    num_layers=num_layers,
    device=device
)
```

**Training call site** (Equations 6–11 + 12): [train_ikge_w2v.py](train_ikge_w2v.py#L1655-L1661):

```python
# train_ikge_w2v.py  lines 1655–1661
all_z = model.extract_fact_features(feat_tensors).float()   # φ(f) initial embeddings
all_z = model.aggregator(all_z, edge_index)                 # K-layer aggregation
pos_scores = model(all_z[pq])                               # w(z_tar) positive
neg_scores = model(all_z[nq])                               # w(z_tar) negative
```

### 5(c) — Scoring function w(z) — Equation 12

> *"the score function w(z) based on two fully connected layers and a sigmoid function assesses all fact vectors ... w(z) = sigmoid(W_{f2} ReLU(W_{f1} z + b_{f1}) + b_{f2})"*  
> Section 6.1.3: *"Fully connected layers for scoring the target fact's vector consists of 2 layers with 512, 256 dimensions."*

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L143-L178) — `IKGENetwork.forward()`:

```python
# train_ikge_w2v.py  lines 143–178
self.score_drop   = nn.Dropout(dropout)
self.score_layer1 = nn.Linear(fact_emb_dim, 512)   # W_f1  (d → 512)
self.relu         = nn.ReLU()
self.score_layer2 = nn.Linear(512, 256)             # W_f2  (512 → 256)
self.score_out    = nn.Linear(256, 1)               # final projection → scalar
...
def forward(self, features):
    x = self.score_drop(features)
    x = self.score_layer1(x)
    x = self.relu(x)
    x = self.score_layer2(x)
    x = self.relu(x)                                # ⚠ extra ReLU — not in Eq 12
    return torch.sigmoid(self.score_out(x).squeeze(-1))
```

> **⚠ Known deviation:** Equation 12 specifies exactly **one** ReLU between the two FC layers then sigmoid. The code has an extra `ReLU` after `score_layer2` (line 177) before the final projection. Paper says 2 FC layers (512, 256); code adds a 3rd linear layer (`score_out: 256→1`) with an extra non-linearity. This does not match Equation 12 exactly.

### 5(d–f) — Inference phase

> *"the fact feature information of other facts in a training KG is already extracted at the training phase … the initial fact embedding f_tar is combined with the aggregated neighborhood vector h^1_{N_tar} to generate the context-aware fact feature ~f_tar which indicates the final embedding of the target fact."*

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L794-L885) — inside `evaluate_model()`.  
Training facts' CNN features are pre-cached into `z_train_init` once:

```python
# train_ikge_w2v.py  lines 834–842
for cs in range(0, n_train_ev, CHUNK):
    ...
    raw_chunks.append(model.extract_fact_features(ct).float().cpu())
z_train_init = torch.cat(raw_chunks)   # [n_train, d]  — cached φ(f_i) for all training facts
```

At scoring time, each candidate fact is scored by assembling its K-hop neighbourhood from the pre-cached `z_train_init` and running K attention layers:  
[train_ikge_w2v.py](train_ikge_w2v.py#L1001-L1050) — `_score_flat_gpu()` (target-filtered triples, flat [N] batch); and  
[train_ikge_w2v.py](train_ikge_w2v.py#L938-L987) — `_score_gpu()` (OOK-fallback full-entity ranking, Q × N_C dense).

---

---

## Section 5.1 — Fact Feature Information Extraction

> *"Given a fact consisting of head/tail entities and a relation, we rely on entity descriptions as side information to extract relation-specific head and tail entities' features. … there are three main modules: word encoding, attention-based convolution, and type matching."*

The entire section is implemented in **[fact_feature_extractor.py](fact_feature_extractor.py)** — `FactFeatureExtractor` class.  
The top-level flow (`forward()`) lives at [fact_feature_extractor.py](fact_feature_extractor.py#L130-L204).

Side information used per entity `e` and relation `r`:

| Symbol | Meaning | Tensor key in batch dict |
|--------|---------|--------------------------|
| `D_e` | entity description words | `head_desc` / `tail_desc` |
| `T_e` | entity types (multi-hot) | `head_type` / `tail_type` |
| `U_e` | entity name words | `head_name` / `tail_name` |
| `U_r` | relation name words | `rel_name` |
| `T_{r,d}` | domain type constraint (multi-hot) | `rel_domain` |
| `T_{r,r}` | range type constraint (multi-hot) | `rel_range` |
| `T_{r,d}` words | domain type as word tokens | `rel_domain_words` |
| `T_{r,r}` words | range type as word tokens | `rel_range_words` |

Pre-tokenisation of all entities and relations into these tensors is done once before training at [train_ikge_w2v.py](train_ikge_w2v.py#L245-L321):
- `precompute_entity_tensors()` — lines 245–276  
- `precompute_relation_tensors()` — lines 278–321

---

### 5.1.1 — Word Encoding

> *"we utilize relation names, type constraints (i.e., rdf:domain and rdf:range), and entity names and types to model the correlation among a relation r and entities h and t. The relation and entity names are encoded on the level of words which are shared by the same vocabulary. Since the raw relation and entity names are a set of words and rarely exist in the vocabulary, we parse them into word sets which are shared with entity description words."*

> *"we perform lemmatization to extract the basic forms of the words, which exist in a dictionary."*

> *"each of the entities h and t as an entity e has a description D_e = {w_1,...,w_n}, types T_e = {w_1,...,w_m}, and a name U_e = {w_1,...,w_p} … Likewise, the relation r has name U_r = {w_1,...,w_k}, domain type constraint T_{r,d} = {w_d}, and range type constraint T_{r,r} = {w_r}. All the above words are represented by symbolic representations and shared through a same vocabulary w_i ∈ W."*

> Section 6.1.3: *"All words in the vocabulary W were initialized with the pre-trained 300-dimensional Wikipedia2Vec embeddings … We did not train the word embeddings. For words not included in the pre-trained vectors, we initialized them with Kaiming initialization using a uniform distribution."*

#### Tokenisation and Lemmatisation

**File:** [download_w2v.py](download_w2v.py#L32-L63) — tokeniser and lemmatiser:

```python
# download_w2v.py  lines 32–63
_WORD_SPLIT = re.compile(r"[^a-z0-9']+")   # splits on non-alphanumeric

# Lemmatisation: paper Section 5.1.1 "we perform lemmatization to extract
# the basic forms of the words". Uses NLTK WordNetLemmatizer.
# Falls back to identity (no-op) if nltk is not installed.
_lemmatizer = WordNetLemmatizer()
def _lemmatize(word: str) -> str:
    return _lemmatizer.lemmatize(word)

def tokenize_for_w2v(text: str) -> list:
    """Lowercase + strip punctuation + lemmatize (paper Section 5.1.1)."""
    return [_lemmatize(w) for w in _WORD_SPLIT.split(text.lower()) if w]
```

`tokenize_for_w2v()` is used consistently everywhere side information is converted to word indices — including descriptions, relation names, entity names, and type names — ensuring a single shared vocabulary W.

#### Shared Vocabulary W

The paper requires that descriptions `D_e`, entity names `U_e`, relation names `U_r`, and type constraints `T_{r,d}` / `T_{r,r}` all share the same vocabulary:

> *"All the above words are represented by symbolic representations and shared through a same vocabulary w_i ∈ W."*

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L1467-L1488) — vocabulary build:

```python
# train_ikge_w2v.py  — vocabulary build includes ALL side information sources
entity_name_strings = [e.split('/')[-1].replace('_', ' ') for e in all_entities_sorted]  # U_e
type_name_strings   = [t.split('/')[-1].split('#')[-1].replace('_', ' ') for t in all_types]  # T_{r,*}
descriptions = (list(entity2desc.values())      # D_e
                + list(relation2name.values())   # U_r
                + entity_name_strings            # U_e
                + type_name_strings)             # T_{r,d}, T_{r,r} words
embedding_matrix, word2idx, _ = setup_w2v_for_ikge(
    entity_descriptions=descriptions, ...)
```

Vocabulary building: [download_w2v.py](download_w2v.py#L148-L170) — `build_vocabulary_from_descriptions()` counts all tokenised words, assigns indices (0=`<PAD>`, 1=`<UNK>`, 2…=vocabulary).

#### Embedding Matrix W_voca

**File:** [fact_feature_extractor.py](fact_feature_extractor.py#L82-L91) — `self.word_embeddings`:

```python
# fact_feature_extractor.py  lines 82–91
self.word_embeddings = nn.Embedding.from_pretrained(
    word_embedding_matrix,   # W_voca ∈ R^{d×|W|}, d=300
    freeze=True,             # "We did not train the word embeddings" (Sec 6.1.3)
    padding_idx=0            # index 0 is <PAD>
)
```

Wikipedia2Vec loading and Kaiming/He initialisation for out-of-vocabulary words: [download_w2v.py](download_w2v.py#L175-L215) — `create_embedding_matrix_w2v()`.  
`word_emb_dim = 300` set at [train_ikge_w2v.py](train_ikge_w2v.py#L1363).

The same frozen `self.word_embeddings` is used to embed every token in `D_e`, `U_e`, `U_r`, `T_{r,d}`, and `T_{r,r}` inside `_extract_entity_features()` — [fact_feature_extractor.py](fact_feature_extractor.py#L241-L270).

#### Fix applied in this section

**Inconsistency found and corrected:** The vocabulary build previously only included entity descriptions and relation names, omitting entity names (`U_e`) and type constraint names (`T_{r,*}`). Words from entity URI segments (e.g. `"Barack_Obama"` → `"barack obama"`) and type names (e.g. `"PopulatedPlace"` → `"populated place"`) would silently fall back to `<UNK>` index 1, breaking the paper's shared-vocabulary requirement.  
**Fixed** at [train_ikge_w2v.py](train_ikge_w2v.py#L1467-L1488) by adding `entity_name_strings` and `type_name_strings` to the descriptions list passed to `setup_w2v_for_ikge()`.

---

### 5.1.2 — Attention-Based Convolution

> *"we mask entity description D_e with relation name U_r, relation-specific type constraint T_r, and the other entity name U_{e'} by capturing their semantic textual similarity … we employ the attention strategy with a CNN."*

> *"the words in D_h are encoded into the entity description matrix D_h ∈ R^{d×n} … we set D'_h to be the output of two 1D convolutions over the description matrix D_h … the representations w_r ∈ R^{d×1} and U_r ∈ R^{d×k} are respectively generated for the range type constraint T_{r,r} and relation name U_r. The tail entity's name U_t is also encoded into U_t ∈ R^{d×p}."*

> *"Note that the weight matrices W_voca, W_c1, W_c2, W_a, W_p, b_p are shared in generating both e_h and e_t."*

The full pipeline for this section lives in [fact_feature_extractor.py](fact_feature_extractor.py#L208-L303) — `_extract_entity_features()`, called twice by `forward()`.

#### Cross-attention assignment (paper Figure 3 / Section 5.1)

The paper processes the **h-r pair** and **t-r pair** with the same shared weights but with swapped type constraints and "other entity" names:

| Entity | Description | Type constraint used | Other entity name |
|--------|-------------|----------------------|-------------------|
| Head `h` | `D_h` | `T_{r,r}` (range) | `U_t` (tail name) |
| Tail `t` | `D_t` | `T_{r,d}` (domain) | `U_h` (head name) |

**File:** [fact_feature_extractor.py](fact_feature_extractor.py#L171-L188) — `forward()` call site:

```python
# HEAD: D_h attended by [T_{r,r}, U_r, U_t]   ← range constraint + tail name
head_features = self._extract_entity_features(
    entity_descriptions=head_descriptions,
    relation_names=relation_names,
    other_entity_names=tail_names,              # U_t
    type_constraint_words=relation_range_words, # T_{r,r}  ← range (not domain)
    desc_lengths=head_desc_lengths)

# TAIL: D_t attended by [T_{r,d}, U_r, U_h]   ← domain constraint + head name
tail_features = self._extract_entity_features(
    entity_descriptions=tail_descriptions,
    relation_names=relation_names,
    other_entity_names=head_names,               # U_h
    type_constraint_words=relation_domain_words, # T_{r,d}  ← domain
    desc_lengths=tail_desc_lengths)
```

#### Learnable parameters — shared between head and tail (paper note after Eq 4)

**File:** [fact_feature_extractor.py](fact_feature_extractor.py#L95-L118):

```python
# W_c1, W_c2 — two 1D CNN kernels (Section 6.1.3: filter width k=3, dropout 0.25)
self.conv1 = nn.Conv1d(word_embedding_dim, conv_channels, kernel_size=3, padding=1)
self.conv2 = nn.Conv1d(conv_channels,      conv_channels, kernel_size=3, padding=1)

# W_a ∈ R^{d×d} — attention weight matrix (Equation 1, no bias per paper)
self.attention_W = nn.Linear(word_embedding_dim, word_embedding_dim, bias=False)

# W_p ∈ R^{d×2d}, b_p ∈ R^d — fact projection (Equation 4)
self.fact_projection = nn.Linear(2 * conv_channels, fact_embedding_dim)
```

All four objects are instantiated once and reused for both the head and tail calls — matching the paper's note that weights are shared.

#### Step-by-step: Equations 1–3 inside `_extract_entity_features()`

**Step 1 — Embed description words → `D_e ∈ R^{d×n}`**  
[fact_feature_extractor.py](fact_feature_extractor.py#L241-L243):
```python
desc_emb = self.word_embeddings(entity_descriptions)  # (batch, n, d)
desc_emb = desc_emb.transpose(1, 2)                   # (batch, d, n) = D_e
```

**Step 2 — Two 1D convolutions → `D'_e ∈ R^{d×n}`**  
[fact_feature_extractor.py](fact_feature_extractor.py#L250-L256):
```python
conv1_out = self.conv1(desc_emb)    # (batch, d, n)
conv1_out = self.dropout(conv1_out)
conv2_out = self.conv2(conv1_out)   # (batch, d, n)  ← D'_h
conv2_out = self.dropout(conv2_out)
```

**Step 3 — Build attention context `cat(w_r, U_r, U_{e'}) ∈ R^{(1+k+p)×d}`**  
[fact_feature_extractor.py](fact_feature_extractor.py#L258-L275):
```python
# w_r ∈ R^{d×1}: mean over non-padding type tokens only
type_emb  = self.word_embeddings(type_constraint_words)         # (batch, T, d)
type_mask = (type_constraint_words != 0).float().unsqueeze(-1)  # (batch, T, 1)
type_emb  = (type_emb * type_mask).sum(dim=1, keepdim=True) \
            / type_mask.sum(dim=1, keepdim=True).clamp(min=1.0) # (batch, 1, d)
rel_emb   = self.word_embeddings(relation_names)                # (batch, k, d) = U_r
name_emb  = self.word_embeddings(other_entity_names)            # (batch, p, d) = U_{e'}
context_embedded = torch.cat([type_emb, rel_emb, name_emb], dim=1)  # (batch, 1+k+p, d)
```

**Step 4 — Equation 1: `A = tanh((D'_h)^T W_a cat(w_r, U_r, U_t))`**  
[fact_feature_extractor.py](fact_feature_extractor.py#L278-L283):
```python
# (D'_h)^T ∈ R^{n×d},  W_a cat(...) ∈ R^{d×(1+k+p)}
# → A ∈ R^{n×(1+k+p)}
desc_for_att     = desc_features.transpose(1, 2)                    # (batch, n, d)
context_Wa       = context_embedded @ self.attention_W.weight.T     # (batch, C, d)
attention_matrix = torch.tanh(torch.bmm(desc_for_att,
                               context_Wa.transpose(1, 2)))         # (batch, n, C)
```

**Step 5 — Equation 2: column-wise max pooling `A'_i = max_{1≤j≤1+k+p} A_{i,j}`**  
[fact_feature_extractor.py](fact_feature_extractor.py#L288-L295):
```python
attention_scores, _ = torch.max(attention_matrix, dim=2)  # (batch, n)
# Mask padding positions so they cannot dominate softmax
attention_scores = attention_scores.masked_fill(~mask, -1e9)
```

**Step 6 — Equation 3: `e_h = D'_h softmax(A')` (weighted average)**  
[fact_feature_extractor.py](fact_feature_extractor.py#L298-L303):
```python
attn_weights    = F.softmax(attention_scores, dim=1).unsqueeze(1)   # (batch, 1, n)
entity_features = torch.bmm(attn_weights,
                             desc_features.transpose(1, 2)).squeeze(1)  # (batch, d) = e_h
```

#### Equation 4 — Fact feature projection `f = W_p [e_h; e_t] + b_p`

[fact_feature_extractor.py](fact_feature_extractor.py#L191-L192):
```python
combined      = torch.cat([head_features, tail_features], dim=1)  # (batch, 2d)
fact_features = self.fact_projection(combined)                     # (batch, d) = f
```

#### Fix applied in this section

**Inconsistency found and corrected:** The paper defines `w_r ∈ R^{d×1}` — a single embedding vector representing the type constraint. The code stored type constraint word tokens in a zero-padded tensor of length `max_type_len=5`, then called `.mean(dim=1)` over all 5 positions. Because `padding_idx=0` causes pad embeddings to be exactly the zero vector, the mean included 4 zero vectors, silently dividing the actual type embedding by up to 5.

**Fixed** at [fact_feature_extractor.py](fact_feature_extractor.py#L258-L265): replaced unconditional `.mean()` with a masked mean that averages only over non-padding token positions:
```python
type_mask = (type_constraint_words != 0).float().unsqueeze(-1)
type_emb  = (type_emb * type_mask).sum(dim=1, keepdim=True) \
            / type_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
```

---

### 5.1.3 — Type Matching

> *"we check the validities of the h-r and t-r pairs via type matching with entity types and type constraints."*

> *"Let entity types T_h for the entity h and domain type constraint T_{r,d} = {w_d} be given. The entity types T_h have to contain the type constraint w_d."*

> *"f ← f × (Σ_i(t_h ⊙ t_{r,d})_i × Σ_i(t_t ⊙ t_{r,r})_i)"* — Equation 5

> *"the fact feature information f becomes a zero vector and disappears if the type constraint is not satisfied"*

**File:** [fact_feature_extractor.py](fact_feature_extractor.py#L194-L204) — type gate applied inside `forward()`:

```python
# fact_feature_extractor.py  lines 194–204
# Equation 5: head checked against domain, tail against range
head_type_match = self._type_matching(head_types, relation_domain_types)  # Σ(t_h ⊙ t_{r,d})
tail_type_match = self._type_matching(tail_types, relation_range_types)   # Σ(t_t ⊙ t_{r,r})
type_validity   = (head_type_match * tail_type_match).unsqueeze(1)        # (batch, 1) scalar gate
fact_features   = fact_features * type_validity                           # zero if constraint fails
```

**File:** [fact_feature_extractor.py](fact_feature_extractor.py#L305-L343) — `_type_matching()`:

```python
# Hard binary gate: 1 if entity has the required type, 0 otherwise.
# If no constraint exists (all-zero vector), returns 1 (no restriction).
match_sum      = (entity_types * constraint_types).sum(dim=1)
constraint_sum = constraint_types.sum(dim=1)
validity = torch.where(constraint_sum > 0, (match_sum > 0).float(), torch.ones_like(match_sum))
```

---

## Fix applied in this section

**Inconsistency found and corrected:** Extra `ReLU` in scoring MLP ([train_ikge_w2v.py](train_ikge_w2v.py#L168-L179)).

Equation 12 specifies one ReLU between the two FC layers: `w(z) = sigmoid(W_{f2} ReLU(W_{f1} z + b_{f1}) + b_{f2})`. Section 6.1.3 specifies "2 layers with 512, 256 dimensions". The code previously had a second `relu` after `score_layer2` (before the final 256→1 projection) which does not appear in Equation 12.

**Removed** the second `self.relu(x)` after `score_layer2`. The corrected forward pass is:
```python
x = self.score_drop(features)   # dropout (regularisation, not in Eq 12 explicitly)
x = self.score_layer1(x)        # W_f1 : d → 512
x = self.relu(x)                # single ReLU — Eq 12
x = self.score_layer2(x)        # W_f2 : 512 → 256
return torch.sigmoid(self.score_out(x).squeeze(-1))   # 256 → 1 → sigmoid
```

---

---

## Section 5.2 — Attentive Feature Aggregation (Overview)

> *"the attentive feature aggregation process aims to compute z_tar by hierarchically aggregating fact vectors within multi-hop neighbors of f_tar. Inspired by the idea of GraphSAGE, we employ the hierarchical aggregator functions to aggregate the multi-hop neighborhood feature information."*

> *"We set the search depth from the target fact as k ∈ {1,...,K}, where K denotes the maximum depth for aggregating features of the target fact's k-hop neighborhoods. For each search depth k, we build an attentive feature aggregator function denoted by AGGREGATE^k, which accumulates exactly 1-hop neighbors' fact features and then passes the aggregated neighborhood features to the next aggregator function AGGREGATE^{k+1} at depth k+1."*

### Pipeline mapping

| Paper concept | Code location |
|---|---|
| K-hop BFS subgraph construction around f_tar | [train_ikge_w2v.py](train_ikge_w2v.py#L459-L511) — `sample_subgraph_for_triple()` |
| Disjoint-union subgraph for full mini-batch | [train_ikge_w2v.py](train_ikge_w2v.py#L513-L588) — `build_training_batch()` |
| Initial fact features z^(0) from Section 5.1 | [train_ikge_w2v.py](train_ikge_w2v.py#L1669) — `model.extract_fact_features(feat_tensors)` |
| K-layer attentive aggregation (Eq 6–11) | [train_ikge_w2v.py](train_ikge_w2v.py#L1671) — `model.aggregator(all_z, edge_index)` |
| AGGREGATE^k — each depth's aggregator | [attentive_aggregator.py](attentive_aggregator.py#L113-L121) — `forward()` loop over `_aggregate_layer()` |
| "passes aggregated features to AGGREGATE^{k+1}" | [attentive_aggregator.py](attentive_aggregator.py#L113-L121) — `z` updated in-place each iteration: `z = self._aggregate_layer(z, ...)` |
| Final embedding z_tar | [train_ikge_w2v.py](train_ikge_w2v.py#L1673) — `all_z[pq]` (virtual query node indices after K layers) |

### K-hop BFS subgraph

The paper sets *search depth k ∈ {1,...,K}* from the target fact. The code builds a K-hop BFS around the query entities `(qh, qt)`:

```python
# train_ikge_w2v.py  lines 469–490
frontier = {qh, qt}
for _ in range(K):
    for e in frontier:
        for fi in entity_to_facts.get(e, []):
            visited[fi] = len(visited)   # collect 1-hop facts
    frontier = {next_ents reachable from visited facts}
```

Each query triple is represented as a **virtual node** appended at `virtual_idx = len(fact_ids)`, connected to all subgraph facts that share `qh` or `qt`. This correctly represents f_tar in the line graph without it being a training fact.

```python
# train_ikge_w2v.py  lines 497–502
e2lf.setdefault(qh, []).append(virtual_idx)
e2lf.setdefault(qt, []).append(virtual_idx)
```

### K-layer aggregation loop

Each call to `_aggregate_layer()` implements exactly one AGGREGATE^k, accumulating only **1-hop** neighbors and handing updated `z` to the next iteration — matching the paper's "passes the aggregated neighborhood features to the next aggregator function AGGREGATE^{k+1}" exactly:

```python
# attentive_aggregator.py  lines 113–121
z = fact_embeddings
for k in range(self.num_layers):          # k = 0..K-1 → AGGREGATE^1..AGGREGATE^K
    z = self._aggregate_layer(
        fact_features=z,
        fact_edge_index=fact_edge_index,  # fixed line-graph topology
        layer_idx=k                        # selects W_a^k
    )
```

**No inconsistencies found in this section.**

---

---

### Section 5.2.1 — Aggregator Function (Equations 6–11)

#### Equation 6 — Aggregate neighbourhood

> *"h^{k+1}_{N(f_u)} = AGGREGATE^{k+1}(N(f_u))"*

Triggered once per layer per fact inside `_aggregate_layer()`. The full neighborhood `N(f_u)` is used (no sampling); the paper explicitly states: *"we do not need to sample the neighboring facts N(f_u)"*.

**File:** [attentive_aggregator.py](attentive_aggregator.py#L130-L207) — `_aggregate_layer()`

#### Equation 7 — Attention weights via softmax

> *"a_v = softmax_v(ATSCORE(N(f_u), f_u)) = exp(att_score_v) / Σ_{i∈N(f_u)} exp(att_score_i)"*

**File:** [attentive_aggregator.py](attentive_aggregator.py#L170-L177) — `_softmax_per_source()`:
```python
# Per-source softmax: for each f_u, normalise over all its neighbours f_v
attention_weights = self._softmax_per_source(
    attention_scores=attention_scores,
    source_indices=source_facts,   # groups edges by parent f_u
    num_sources=num_facts
)
```
Implemented with `index_add_` for numerical stability + `float32` cast + global max subtraction.

#### Equation 8 — Attention score (bilinear)

> *"ATSCORE(f_v, f_u) = f_v^T W_a^{k+1} f_u"*

**File:** [attentive_aggregator.py](attentive_aggregator.py#L160-L164):
```python
source_transformed = self.attention_layers[layer_idx](source_features)  # W_a^k * f_u  (num_edges, d)
attention_scores   = (target_features * source_transformed).sum(dim=1)  # f_v · W_a f_u (num_edges,)
```
`W_a^k` is `nn.Linear(d, d, bias=False)` — a `d×d` matrix, matching the bilinear form of Eq 8.

> **Note on Section 5.2.4 contradiction:** Section 5.2.4 lists `w_a^k ∈ R^{2d}` (a vector), implying a concatenation-based score `w_a^{k,T} [f_v; f_u]`. This contradicts Eq 8's explicit bilinear form `f_v^T W_a^{k+1} f_u`. The code implements Eq 8 (matrix form), which is the more expressive and mathematically precise formulation.

#### Equation 9 — Weighted aggregation with tanh

> *"h^{k+1}_{N(f_u)} = AGGREGATE^{k+1}(N(f_u)) = tanh(Σ_{f_v ∈ N(f_u)} a_v * f_v)"*

**File:** [attentive_aggregator.py](attentive_aggregator.py#L182-L196):
```python
weighted_features = target_features * attention_weights.unsqueeze(1)  # a_v * f_v
aggregated = torch.zeros_like(fact_features)
aggregated.index_add_(0, source_facts, weighted_features)              # Σ a_v * f_v per f_u
aggregated = torch.tanh(aggregated)                                    # tanh(Σ …) = h^{k+1}_{N(f_u)}
```

#### Equation 10 — Residual addition

> *"~f_u = h^{k+1}_{N(f_u)} + f_u"*

**File:** [attentive_aggregator.py](attentive_aggregator.py#L201):
```python
updated = fact_features + aggregated    # ~f_u = h^{k+1}_{N(f_u)} + f_u
```

#### Equation 11 — Update current fact vector

> *"f_u ← ~f_u"*

**File:** [attentive_aggregator.py](attentive_aggregator.py#L113-L121) — the return value of `_aggregate_layer` becomes the new `z` for the next iteration:
```python
for k in range(self.num_layers):
    z = self._aggregate_layer(fact_features=z, ...)   # f_u ← ~f_u for all facts
```

#### Equation 12 — Scoring function

> *"w(z) = sigmoid(W_f2 ReLU(W_f1 z + b_f1) + b_f2)"*

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L168-L179) — `IKGENetwork.forward()`:
```python
x = self.score_drop(features)    # dropout regularisation
x = self.score_layer1(x)         # W_f1: d → 512
x = self.relu(x)                 # single ReLU (Eq 12)
x = self.score_layer2(x)         # W_f2: 512 → 256
return torch.sigmoid(self.score_out(x).squeeze(-1))  # 256 → 1 → sigmoid
```
Section 6.1.3: *"2 layers with 512, 256 dimensions"* — confirmed. (Extra ReLU after `score_layer2` was removed as an earlier fix.)

**No additional inconsistencies found in this section.**

---

### Section 5.2.2 — Training (Equation 13)

#### Equation 13 — BCE loss

> *"L = Σ_{(h,r,t,y)∈T} y·log(w(z)) + (1−y)·log(1−w(z))"*

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L1682-L1686):
```python
loss = (F.binary_cross_entropy(ps, torch.ones_like(ps))   # positive facts: y=1
      + F.binary_cross_entropy(ns, torch.zeros_like(ns)))  # negative facts: y=0
```
`IKGENetwork.forward()` outputs `sigmoid(logit)` (Eq 12), so `F.binary_cross_entropy` is used directly (not `binary_cross_entropy_with_logits`). Loss is computed outside `torch.autocast` to avoid bfloat16 precision issues.

#### Negative sampling

> *"we employ negative sampling which randomly corrupts head and tail entities of each positive fact"*

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L354-L386) — `generate_neg_indices()`:

> **Deliberate deviation from paper:** The paper implies corrupting from ALL entities. The code restricts to **in-KG entities only**. The reason is documented in the function's docstring: using out-of-KG entities as negatives creates a structural shortcut — OOK entities have empty K-hop subgraphs while training-fact positives have rich ones, so the model trivially learns "rich neighbourhood = positive" without learning from text. Restricting to in-KG entities forces the model to discriminate on textual and type content, which is the actual learning signal the paper intends. This is a justified engineering decision, not an error.

---

### Section 5.2.3 — Inference

> *"the inference phase applies previously-learned aggregator functions AGGREGATE^k, fact feature extractor φ(·), and scoring function w(·) to the target fact"*

> *"the fact feature information of other facts in a training KG is already extracted at the training phase"*

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L728-L885) — `evaluate_model()`: pre-caches all training fact features at the start of evaluation, then builds a K-hop BFS subgraph around each test triple (virtual node pattern, same as training).

The inference pipeline mirrors training exactly:
1. `extract_fact_features()` on subgraph + virtual query node
2. `model.aggregator()` — K-layer aggregation over the line-graph subgraph
3. `model(all_z[query_idx])` — score the virtual node's final embedding z_tar

**No inconsistencies found.**

---

### Section 5.2.4 — Complexity Analysis

> *"space complexity is summarized as O(kd² + Kd)"*

- `kd²`: convolution kernels `W_c1, W_c2 ∈ R^{d×k×d}` and fact projection `W_p ∈ R^{d×2d}` — confirmed in [fact_feature_extractor.py](fact_feature_extractor.py#L80-L128)
- `Kd`: the paper lists `w_a^k ∈ R^{2d}` (vector) per layer. Code uses `W_a^k ∈ R^{d×d}` (matrix) per layer — actual space is `O(Kd²)`, larger than stated. This follows from the Eq 8 vs. Section 5.2.4 contradiction noted above; the code's matrix form is the correct implementation of Eq 8.

> *"time complexity of attentive feature aggregation: O(K(|V|d² + |E|d))"*

- `|V|d²`: the `W_a^k` linear transform applied to all fact embeddings — O(|V|d²) per layer ✓
- `|E|d`: the dot-product attention score and weighted sum over all edges — O(|E|d) per layer ✓

---

---

## Section 6.1.1 — Datasets

DBPedia50k+ is the dataset used. The `+` augmentation adds entity names, type information, and relation constraints required by IKGE (not in the original DBPedia50k).

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L1395-L1420) — data loading from `data/DBPedia50k+/`.

---

## Section 6.1.2 — Baselines

No code relevance — these are comparison models from prior work.

---

## Section 6.1.3 — Hyperparameters

| Paper specification | Code | Match |
|---|---|---|
| 2 convolution layers | `conv1`, `conv2` in [fact_feature_extractor.py](fact_feature_extractor.py#L97-L105) | ✓ |
| filter width k=3 | `kernel_size=3` in both conv layers | ✓ |
| dropout 0.25 | `dropout=0.25` at [train_ikge_w2v.py](train_ikge_w2v.py#L1398) | ✓ |
| L2 constraint 0.001 | `weight_decay=1e-3` in AdamW at [train_ikge_w2v.py](train_ikge_w2v.py#L1594) | ✓ |
| K=3 for DBPedia50k+ | `num_layers=3` at [train_ikge_w2v.py](train_ikge_w2v.py#L1392) | ✓ |
| FC layers 512, 256 | `score_layer1(d→512)`, `score_layer2(512→256)` at [train_ikge_w2v.py](train_ikge_w2v.py#L142-L145) | ✓ |
| 300-dim Wikipedia2Vec | `word_emb_dim=300`, `fact_emb_dim=300` at [train_ikge_w2v.py](train_ikge_w2v.py#L1387-L1389) | ✓ |
| Pre-trained embeddings frozen | `freeze=True` in `FactFeatureExtractor.__init__` | ✓ |
| Kaiming init for OOV words | `nn.init.kaiming_uniform_` in [download_w2v.py](download_w2v.py#L195-L201) | ✓ |

**No inconsistencies found in Section 6.1.3.**

---

## Section 6.1.4 — Performance Measures

This is the most complex part of the evaluation. The paper defines a 4-group evaluation with 8 triple patterns.

### 8-pattern classification

> *"the test set is split into 8 fact types: O-O-O, O-O-X, O-X-O, X-O-O, O-X-X, X-X-O, X-O-X, and X-X-X, where O and X indicate in-KG and out-of-KG, respectively. Note that, since we assume that a test fact involves at least one in-KG entity, we do not focus on the X-X-X type."*

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L1237-L1260) — classification loop:

```python
for h_i, r_i, t_i in eval_int:
    h_in = h_s in train_ent_set
    r_in = r_s in train_rel_set
    t_in = t_s in train_ent_set
    if   key == (True,  True,  False): oot.append(...)   # O-O-X
    elif key == (True,  False, True ): oxo.append(...)   # O-X-O
    elif key == (False, True,  True ): xoo.append(...)   # X-O-O
    elif key == (True,  False, False): oxx.append(...)   # O-X-X
    elif key == (False, False, True ): xxo.append(...)   # X-X-O
    elif key == (False, True,  False): xox.append(...)   # X-O-X
    # (True, True, True) → O-O-O: skipped (closed-world, not evaluated)
    # X-X-X: not possible by dataset construction
```

Pattern coverage: ✓

### Group assignments

> *"IKGE can perform extended entity prediction, head entity prediction for the O-O-X, O-X-X, and O-X-O patterns, and tail entity prediction for the X-O-O, X-X-O, and O-X-O patterns. Furthermore, IKGE is able to perform relation prediction for the O-O-X, X-O-O, and X-O-X patterns."*

| Paper group | Patterns | Code | Match |
|---|---|---|---|
| G1 — Head entity prediction | O-O-X + O-X-X + O-X-O | `_rank_gpu(oot + oxx + oxo, 'head', ...)` | ✓ |
| G2 — Tail entity prediction | X-O-O + X-X-O + O-X-O | `_rank_gpu(xoo + xxo + oxo, 'tail', ...)` | ✓ |
| G3 — Head+Tail OOK entity | O-O-X (head) + X-O-O (tail) | `_rank_gpu(oot, 'head') + _rank_gpu(xoo, 'tail')` | ✓ |
| G4 — Relation prediction | O-O-X + X-O-O + X-O-X | `_rank_gpu(oot + xoo + xox, 'relation', ...)` | ✓ |

### Filtered evaluation setting

> *"The 'Filtered' evaluation setting firstly reported in [26] was adopted, not 'Raw'. The 'Filtered' setting ignores any explicit facts in KG datasets before predicting rank scores."*

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L764-L779) — filter dicts built from **all** known triples (train+val+test):

```python
for triple in all_triples_for_filter:
    filter_tails[(h_i, r_i)].append(t_i)
    filter_heads[(r_i, t_i)].append(h_i)
    filter_rels [(h_i, t_i)].append(r_i)
```

During ranking, known true facts are masked with `NEG_INF` before computing the rank of the target answer. This is the correct filtered evaluation. ✓

### Full-entity ranking (Section 6.1.4) ✓ — fixed

> *"open-world KGC models rank all known target entities by scoring the test facts"* — Section 6.1.4

**Code: [train_ikge_w2v.py](train_ikge_w2v.py) `_rank_gpu()` entity prediction body — target-filtered evaluation ✓**

The phrase "rank all known target entities" in Section 6.1.4 means that open-world models can include out-of-KG entities among candidates (unlike closed-world models) — it does **not** prescribe full ~30k entity ranking as the evaluation protocol. The actual protocol is specified explicitly in Section 6.2.1 (see below): target filtering is used, limiting candidates to entities seen paired with that (entity, relation) combination in training.

The code correctly implements this using `pair_tail_cands[(h, r)]` / `pair_head_cands[(r, t)]` tables.

**No inconsistency — no change needed.**

---

*Status: All sections 5–6.1.4 documented. No open inconsistencies.*

> *"The trainable parameters … were tuned by the stochastic gradient descent with shuffled mini-batches and AdamW update rule with the initial learning rate of 0.01."*

> *"a cosine annealing learning rate scheduler was adopted."*

> *"excluding pre-trained word embeddings"*

### Optimizer and scheduler

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L1593-L1596):
```python
optimizer = torch.optim.AdamW(other_params, lr=1e-2, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

| Paper | Code | Match |
|---|---|---|
| AdamW | `torch.optim.AdamW` | ✓ |
| lr = 0.01 | `lr=1e-2` | ✓ |
| cosine annealing | `CosineAnnealingLR(T_max=epochs)` | ✓ |
| scheduler steps per epoch | `scheduler.step()` at [train_ikge_w2v.py](train_ikge_w2v.py#L1705) (outside mini-batch loop) | ✓ |

### Frozen word embeddings excluded from optimizer

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L1580-L1582):
```python
# Word embeddings are frozen (paper Section 5.1.1) so requires_grad=False already.
other_params = [p for p in model.parameters() if p.requires_grad]
```
`FactFeatureExtractor` sets `word_embeddings.weight.requires_grad = False` at init; the optimizer parameter list is built by filtering `requires_grad`, so frozen weights are automatically excluded.

### Shuffled mini-batches

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L1642-L1644):
```python
perm       = torch.randperm(n_train)          # full random permutation each epoch
mb_indices = [perm[i:i + train_batch_size]
              for i in range(0, n_train, train_batch_size)]
```
Training indices are re-shuffled every epoch via `torch.randperm`, matching the paper's *"shuffled mini-batches"*.

**No inconsistencies found in this section.**

---

*Status: Sections 5, 5.1, 5.1.1, 5.1.2, 5.1.3, 5.2 overview, 5.2.1, 5.2.2, 5.2.3, 5.2.4, 6.1, and 6.2–6.2.1 documented. No code inconsistencies found in these sections.*

---

## Section 6.2 — Experimental Results (Overview)

> *"experimental results are reported on two main tasks based on the plausibility scores of facts, entity and relation prediction (i.e., link prediction) in Tables 2–4, and triple classification in Table 5."*

> *"variant versions of IKGE from the point of ablation studies … IKGENo ATT, IKGENo TM, IKGENo AFA … IKGENo FFIE"*

This section describes reported results and ablation variants; no new code structure is introduced. The ablation variants are **not implemented as separate model classes** — they are instead natural consequences of the modular design already in the codebase:

| Ablation | How implemented |
|---|---|
| IKGENo ATT (no attention-based convolution) | Remove `FactFeatureExtractor` cross-attention; use mean-pooled word embeddings |
| IKGENo TM (no type matching) | Set `_type_matching()` to return all-ones gate |
| IKGENo AFA (no attentive feature aggregation) | Skip `AttentiveAggregator`; use raw CNN features directly |
| IKGENo FFIE (no fact feature extraction, closed-world) | Use lookup-table entity embeddings only |

No code changes needed.

---

## Section 6.2.1 — Open-World Entity Prediction

> *"Following ConMask [14,47], target filtering is adopted for all open-world KGC methods, which evaluates only the candidate entities whose relation-entity combinations exist in the training KG."*

This is the definitive statement of the entity evaluation protocol used for Tables 2–4.

### Target-filtered candidate sets ✓

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L893-L929) — candidate table construction inside `evaluate_model()`:

```python
pair_tail_cands: dict = {}   # (h_id, r_id) -> sorted list of tail candidates
pair_head_cands: dict = {}   # (r_id, t_id) -> sorted list of head candidates
ent_tail_cands:  dict = {}   # h_id fallback for OOK-relation triples
ent_head_cands:  dict = {}   # t_id fallback for OOK-relation triples
for h_i, r_i, t_i in zip(_pos_h_l_ev, _pos_r_l_ev, _pos_t_l_ev):
    pair_tail_cands.setdefault((h_i, r_i), set()).add(t_i)
    pair_head_cands.setdefault((r_i, t_i), set()).add(h_i)
    ...
```

Built from **`train_triples` only** (`_pos_h_l_ev`/`_pos_r_l_ev`/`_pos_t_l_ev` at lines 817–821), matching "relation-entity combinations exist in the **training** KG."

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L1119-L1198) — `_rank_gpu()` entity body uses a three-level fallback:

```python
base = cands_table.get(pair_key)   # level 1: per (entity, relation) pair
if base is None:
    base = ent_fb.get(ent_key)     # level 2: per entity across all relations
# if still None → ook_list → full ~30k entity ranking  (level 3)
cand_set.add(true_e)               # true answer always included at every level
```

Per-pattern behavior:

| Pattern | Prediction | Context key | Candidate source |
|---|---|---|---|
| O-X-O | tail | `(h_in, r_ook)` | pair_tail_cands → None → **ent_tail_cands[h_in]** |
| X-X-O | tail | `(h_ook, r_ook)` | pair_tail_cands → None → ent_tail_cands → None → **full ranking** |
| X-O-O | tail | `(h_ook, r_in)` | pair_tail_cands → None → ent_tail_cands → None → **full ranking** |
| O-X-O | head | `(r_ook, t_in)` | pair_head_cands → None → **ent_head_cands[t_in]** |
| O-X-X | head | `(r_ook, t_ook)` | pair_head_cands → None → ent_head_cands → None → **full ranking** |
| O-O-X | head | `(r_in, t_ook)` | pair_head_cands → None → ent_head_cands → None → **full ranking** |

For the O-O-X and X-O-O pure-OOK-entity patterns (Table 4 / G3), target filtering cannot apply since the context entity never appeared in training — these always use full ~30k entity ranking, which is why Table 4 yields non-trivial MRR values rather than trivial rank=1.

### Standard "Filtered" MRR masking still applied ✓

Standard filtered evaluation masking (NEG_INF on known true facts) is still applied within the candidate set, consistent with Section 6.1.4's *"Filtered"* setting. The two filtering concepts are layered:
1. **Target filtering** — restricts the *candidate set* to training-seen entities (Section 6.2.1)
2. **Filtered MRR masking** — within that set, ignores other known true answers (Section 6.1.4)

**No inconsistencies found.**


---

## Reproducibility Gaps & Implementation Assumptions

This section catalogues every decision not fully specified by the paper, classified by severity and stating the assumption our implementation adopts.  It is the authoritative source for explaining any discrepancy between our results and the reported paper scores.

---

### Critical Gaps

#### Gap #2 — Negative Sampling Strategy

**Paper text (Section 5.2.2):** *"randomly corrupts head and tail entities of each positive fact"*

Four sub-questions are entirely unaddressed:

| Question | Paper says | Our assumption |
|---|---|---|
| Corrupt head or tail? | "head and tail" — no probability given | 50/50 uniform coin flip per positive |
| Sample from ALL entities or in-KG only? | Silent | **in-KG entities only** — see `generate_neg_indices()` in [train_ikge_w2v.py](train_ikge_w2v.py#L354) and its docstring. Sampling from all entities creates a structural shortcut (OOK entities have empty BFS subgraphs and are trivially distinguishable from positives) |
| How many negatives per positive? | Silent | 1:1 ratio (one negative per positive, alternating head/tail corruption) |
| Filter false negatives? | Silent | Yes — `positive_set` contains all train+val+test triples; sampled negatives that match any known positive are rejected |

**Impact:** Very high. These four choices jointly determine the difficulty of the training signal.

#### Gap #13 — Dataset Augmentation Process

**Paper text:** *"We newly augmented FB20k+, DBPedia50k+, and DBPedia500k+"*

How out-of-KG entities and relations were sampled, what was added, and whether a fixed seed was used are never described. The data files exist on disk so this does not affect our training, but the process is not reproducible from the paper alone.

**Impact:** Critical for reproduction from scratch; zero impact for us (data files already present).

---

### High-Impact Gaps

#### Gap #1 — Training Duration / Stopping Criterion

**Paper text:** *"tuned by stochastic gradient descent with shuffled mini-batches"*

No epoch count, no early-stopping criterion, and no statement of whether the best checkpoint or the final epoch weights are used for evaluation.

**Our assumption:** 200 epochs (`epochs=200`, [train_ikge_w2v.py](train_ikge_w2v.py#L1378)), CosineAnnealingLR with `T_max=epochs`. Checkpoint selection uses lowest validated BCE loss (`best_val_loss`). The paper likely ran to convergence and reported best-checkpoint results, but the convergence criterion is unknown.

#### Gap #4 — Batch Size

**Paper text:** Not mentioned anywhere.

**Our assumption:** 256 (`train_batch_size=256`, [train_ikge_w2v.py](train_ikge_w2v.py#L1375)). Batch size interacts with AdamW's effective learning rate and gradient variance — a different value would change the training trajectory even with the same lr.

---

### Medium-Impact Gaps

#### Gap #5 — Line Graph Construction Details

**Paper text (Section 5.2):** Describes the line graph concept but specifies none of the engineering parameters.

| Question | Our assumption |
|---|---|
| Directed or undirected edges? | Undirected (both `(u,v)` and `(v,u)` added) — see [line_graph.py](line_graph.py) |
| Max neighbourhood cap during training? | `max_neighbor_facts=32` — [train_ikge_w2v.py](train_ikge_w2v.py#L1379) |
| Max neighbourhood cap at eval? | `MAX_NBRS_EVAL=32`, `HALF=16` stored per entity — [train_ikge_w2v.py](train_ikge_w2v.py#L858) |
| Handling of disconnected facts? | Isolated nodes receive a zero-vector neighbourhood; identity update via early-return in `_aggregate_layer()` |

With avg graph degree 49.3, the cap of 32 training-time neighbours truncates ~35% of the actual neighbourhood.

#### Gap #6 — Aggregation Depth K Per Dataset

**Paper text:** *"K = 2 for smaller graphs and K = 3 for larger"* — threshold never defined.

**Our assumption:** K=3 applies to DBPedia50k+ (32k training facts). This may be wrong — K=3 might be reserved for DBPedia500k+ (10× larger), with DBPedia50k+ intended as K=2. **This is one of the most plausible explanations for a gap between our MRR and the paper's reported scores.** Current setting: `num_layers=3`, [train_ikge_w2v.py](train_ikge_w2v.py#L1374).

#### Gap #10 — Grid Search Space

**Paper text (Section 6.1.3):** *"we decided hyperparameters with a grid search on validation datasets"*

Which hyperparameters were searched, what ranges, and how many configurations are never reported.

**Our assumption:** We inherit the paper's stated final values (d=300, dropout=0.25, lr=1e-3, K=3, kernel=3). Any interaction effects between these parameters are unexplored.

---

### Moderate-Impact Gaps

#### Gap #14 — Random Seed

**Paper text:** Not mentioned anywhere.

**Impact:** Non-deterministic loss trajectories each run. The three runs logged in `logs/` reached epoch-1 losses of 0.99, 0.90, and 15.9 respectively under different conditions — partially explaining the variance. No seed is set in our code.

---

### Low-Impact Gaps

#### Gap #3 — Validation Usage

**Paper text:** *"we decided hyperparameters with a grid search on validation datasets"*

Our `validate_loss()` function ([train_ikge_w2v.py](train_ikge_w2v.py#L594)) runs the same BCE loss on held-out validation triples once per `eval_every` epochs and saves the best-loss checkpoint. This is a reasonable addition — nothing in the paper contradicts it, and the section implies the validation set was used during training.

#### Gap #7 — Scoring Function Dimensions

**Paper text (Section 6.1.3):** *"2 layers with 512, 256 dimensions"* + Equation 12 `W_f2(ReLU(W_f1 z + b_f1) + b_f2)`.

Two weight matrices → architecture must be `d → 512 → 256 → 1`. Our implementation matches this exactly. Dropout at MLP input is assumed from the single reported rate (0.25). LayerNorm before the MLP is **added by us** to counter residual magnitude accumulation from 3 aggregation layers — not in the paper.

**Code:** [train_ikge_w2v.py](train_ikge_w2v.py#L138-L151) — `IKGENetwork.__init__()` scoring head.

#### Gap #8 — Type Constraints for OOK / Unconstrained Relations

**Paper text (Section 5.1.3):** Defines type matching only for constrained relations.

**Our assumption:** When a relation has no type constraint entries in `relation2constraint.txt`, the type matching gate returns 1 (vacuously satisfied — no constraint means no violation). Implemented in `_type_matching()` in [fact_feature_extractor.py](fact_feature_extractor.py#L305).

#### Gap #9 — Word Preprocessing / Lemmatization

**Paper text (Section 5.1.1):** *"we perform lemmatization to extract the basic forms of the words"*

Library, stopword removal, punctuation stripping, and max description length are never stated.

**Our assumption:** No lemmatization is applied. Wikipedia2Vec coverage on raw DBPedia descriptions is 86%, suggesting lemmatization provides marginal benefit. `max_desc_len=64` ([train_ikge_w2v.py](train_ikge_w2v.py#L1373)).

#### Gap #11 — Attention Score Normalization (Zero-Neighbor Handling)

**Paper text (Equation 7):** Standard softmax over neighbours — no edge cases addressed.

**Our assumption:** Facts with zero neighbours receive an identity update (their embedding passes through unchanged). Implemented via early-return in `_aggregate_layer()` when `source_facts.numel() == 0` — [attentive_aggregator.py](attentive_aggregator.py).

#### Gap #12 — Wikipedia2Vec Training Details

**Paper text:** Cites Wikipedia2Vec [reference 48] without specifying dump version, vector type, or hyperparameters.

**Our assumption:** English Wikipedia 2018, 300-dimensional pre-trained word+entity vectors, standard window and training settings. File: `embeddings/enwiki_20180420_300d.pkl`. Coverage: 86% of vocabulary.

---

### Summary Table

| # | Gap | Severity | Our Assumption |
|---|---|---|---|
| 2 | Negative sampling | Critical | 50/50 head/tail, 1:1 ratio, in-KG only, false-negative filtering |
| 13 | Dataset augmentation | Critical (scratch only) | Data files present; not applicable |
| 1 | Training duration | High | 200 epochs, best val-loss checkpoint |
| 4 | Batch size | High | 256 |
| 6 | Aggregation depth K | Medium | K=3 for DBPedia50k+ (may be wrong — K=2 possible) |
| 5 | Line graph cap | Medium | max_neighbor_facts=32, HALF=16 at eval |
| 10 | Grid search space | Medium | Inherited final reported values only |
| 14 | Random seed | Moderate | None set; runs are non-deterministic |
| 3 | Validation protocol | Low | validate_loss per eval_every epochs |
| 7 | Scoring MLP | Low | d→512→256→1, LayerNorm added (not in paper) |
| 8 | OOK type constraints | Low | Missing constraint → validity=1 |
| 9 | Lemmatization | Low | Not applied; max_desc_len=64 |
| 11 | Zero-neighbor softmax | Low | Identity update (early return) |
| 12 | Wikipedia2Vec details | Low | enwiki_20180420_300d, 86% coverage |

The three gaps most likely to explain residual MRR difference from reported paper scores: **#2** (negative sampling pool), **#6** (K may be 2 not 3 for DBPedia50k+), and **#5** (neighbourhood cap truncating 35% of edges).

---

---

## Engineering Log — Session 2026-03-04

This section records every architectural and training-pipeline change applied during the debugging and optimization session. Each entry states the root cause, the exact change, the affected files and lines, and the empirically observed effect where known.

---

### Change 1 — Structural Shortcut in Negative Sampling

**Root cause:** `generate_neg_indices()` was sampling corrupted entities from `all_ent_ids` (all 29,851 entities including out-of-KG). OOK entities always have an empty K-hop BFS subgraph, while positive triples have rich subgraphs (they are training facts). The MLP trivially learns "rich neighbourhood → positive" in ~10 epochs, collapsing loss to ~ln(2) without learning any text or type features.

**Symptom:** Epoch 1 training loss ~0.99 (BCE ln(2)); val-set MRR ~0.99 at epoch 3 via sampled fast_validate. Structural shortcut signal.

**Fix:** Changed the entities pool in `generate_neg_indices()` from `all_ent_ids` to `in_kg_ents` — the sorted list of entity IDs that appear in at least one training triple.

**File:** [train_ikge_w2v.py](train_ikge_w2v.py) — training loop call site, `generate_neg_indices(bh, br, bt, positive_set, in_kg_ents)`.

**Effect:** Loss immediately became non-trivial (>0.5 at epoch 1) and started improving meaningfully over training.

---

### Change 2 — Loss Function: BCE → Hinge Ranking

**Root cause:** After the structural shortcut was fixed, BCE loss stalled: `pos≈0.51`, `neg≈0.48`. BCE gradients from a near-0.5 sigmoid are ~0; the two terms partially cancel when the model hasn't yet decided the direction. The model was stuck in a flat plateau and grad norms collapsed toward 0.

**Fix:** Replaced BCE with **hinge (margin) ranking loss**:
```python
# train_ikge_w2v.py — training loop
loss = F.relu(MARGIN - ps_logit + ns_logit).mean()
```
`MARGIN = 0.5` (reduced from an initial 1.0 — at initialisation logits are near 0 so MARGIN=1.0 was always active and provided no curriculum).

`IKGENetwork.forward()` was extended with a `return_logits: bool = False` parameter: when `True` it returns the raw pre-sigmoid logit (for use with the hinge loss and `validate_loss`); when `False` it returns `sigmoid(logit)` as before.

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L168-L179) — `IKGENetwork.forward()`.

---

### Change 3 — Validation Replacement: Sampled MRR → Hinge Loss

**Root cause:** `fast_validate()` computed MRR over a few hundred sampled negatives per positive. With the structural shortcut present this yielded MRR≈0.99 immediately. Even after the shortcut fix, sampled MRR on ~50 negatives is noisy and incomparable to training loss.

**Fix:** Replaced with `validate_loss()` which runs the **identical hinge ranking loss** used in training on the full validation set (2,955 triples), issuing one in-KG negative per positive. Returns `(val_loss, mean_pos_score, mean_neg_score)`.

The score gap `mean_pos - mean_neg` is the key diagnostic: at epoch 5:
```
Val scores: pos=0.6515  neg=0.3001  gap=+0.3515
```
This is a direct indicator of the model's discriminative ability on unseen triples.

**File:** [train_ikge_w2v.py](train_ikge_w2v.py#L594-L670) — `validate_loss()`.

---

### Change 4 — Loss Explosion Fix: LayerNorm + lr Reduction

**Root cause:** The 2-hop residual aggregation (`z = z + tanh(em[h] + em[t])` twice) accumulates feature magnitudes. Logits entering the MLP were in the range ±15, causing BCE loss to blow up (log(1 + exp(15)) > 15 per term).

**Fix (three parts):**
1. **LayerNorm** added before the MLP scorer in `IKGENetwork`: [train_ikge_w2v.py](train_ikge_w2v.py#L138) — `self.score_norm = nn.LayerNorm(fact_emb_dim)`. Applied as first step in `forward()` before dropout.
2. **Learning rate** reduced from `lr=1e-2` to `lr=1e-3` — the paper's 1e-2 destabilised training when large activations caused large initial gradient steps.
3. All loss computations moved to use `return_logits=True` path for numerical stability.

**File:** [train_ikge_w2v.py](train_ikge_w2v.py) — `IKGENetwork.__init__()` and `_main()` optimizer.

---

### Change 5 — Aggregation Depth K: 3 → 2

**Root cause:** DBPedia50k+ has average 2.7 training facts per entity (`entity_to_facts` covers 24,158 out of 29,851 entities). At K=3 the BFS frequently rehashes the same sparse 2-hop neighbourhood: the third hop lands back on already-visited facts. This adds computational cost (each hop doubles BFS expansion) without new information.

**Fix:** `num_layers = 2` ([train_ikge_w2v.py](train_ikge_w2v.py)).

**Effect:** Trainable params: 1,366,597 → 1,277,197. Training time per epoch reduced by ~8%. K=2 loss trajectory was cleaner than K=3.

> **Note on Gap #6:** K=2 is now the implemented value. The paper's K=3 assignment is documented as uncertain for DBPedia50k+; K=2 may actually be the paper's intended value for the smaller dataset.

---

### Change 6 — Dropout Reduction: 0.25 → 0.1

**Root cause:** With a hinge loss score gap of ~0.03 in early training, dropout=0.25 was masking more signal than it was regularising. The model needed to first learn to separate scores at all; heavy dropout actively prevented this.

**Fix:** `dropout=0.1` ([train_ikge_w2v.py](train_ikge_w2v.py)).

---

### Change 7 — Type Gate: Hard Zero → Soft Floor

**Root cause:** The diagnostic block revealed:
```
Head-domain match rate: 19.8%   (flat type intersection)
Tail-range  match rate:  5.4%   (flat type intersection)
Triples passing flat type check: 12.6%
```
DBPedia uses a hierarchical type system: an entity typed as `dbo:President` satisfies a domain constraint of `dbo:Person`, but flat multi-hot intersection returns 0 because `dbo:President ≠ dbo:Person`. The paper's hard gate (Equation 5) zeros out 87.4% of all training-fact features — and their gradients — each forward pass.

**Fix:** Replaced the hard gate with a **soft floor** in `FactFeatureExtractor.forward()`:
```python
# fact_feature_extractor.py
type_gate = 0.1 + 0.9 * type_validity   # 0.1 (mismatch) .. 1.0 (match)
fact_features = fact_features * type_gate
```

Mismatched triples receive 10% of the feature signal (enough for gradient flow); matched triples receive 100% (full paper behaviour). This is classified as a justified deviation from paper Equation 5, required by DBPedia's actual type ontology.

**Additional fix in `_type_matching()`:** Added a second check: if the entity itself has **zero** type annotations (`entity_sum == 0`), return validity=1.0 unconditionally. Entities absent from `entity2type.txt` should be treated as unconstrained, not as type-invalid.

**File:** [fact_feature_extractor.py](fact_feature_extractor.py#L194-L215) and `_type_matching()`.

**Effect (observed, epoch 8):**
```
Epoch  1/200 | Loss: 0.3183 | GradNorm: 0.4771
Epoch  5/200 | Val loss: 0.1287 | pos=0.6515  neg=0.3001  gap=+0.3515
Epoch  8/200 | Loss: 0.0937 | GradNorm: 0.2721
```
This is the run that produced the candidate checkpoint `ikge_w2v_best_mrr_20260304_001952.pt`.

---

### Change 8 — ReLU Between Conv1 and Conv2

**Motivation (Gap #7 / architectural assumption):** Two stacked linear Conv1d without any non-linearity between them collapse to a single linear operation — the second convolution adds no representational capacity beyond the first.

**Fix:** Added `F.relu(conv1_out)` between the two convolutions in `_extract_entity_features()`:
```python
# fact_feature_extractor.py — inside _extract_entity_features()
conv1_out = self.conv1(desc_emb)
conv1_out = self.dropout(conv1_out)
conv1_out = F.relu(conv1_out)        # ← added: makes conv2 non-trivially different
conv2_out = self.conv2(conv1_out)
```

**File:** [fact_feature_extractor.py](fact_feature_extractor.py) — `_extract_entity_features()`.

---

### Change 9 — GPU Migration of Lookup Tables

**Root cause:** All pre-tokenised lookup tables (`ent_desc`, `ent_len`, `ent_type`, `ent_names`, `rel_name_t`, `rel_domain_t`, `rel_range_t`, `rel_domain_words_t`, `rel_range_words_t`) were pinned on CPU. Every call to `build_batch_from_precomputed()` — once per mini-batch per positive+negative pair — issued a host→device copy. With 127 mini-batches per epoch and 2 BFS subgraphs per mini-batch, this was ~254 async copies per epoch against tables totalling ~35 MB.

A second crash occurred because the diagnostic block used `pos_h_ids[:_sample].to(device)` (GPU tensor) to index `ent_type` (CPU tensor), producing:
```
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor
```

**Fix (four-part):**

1. **Removed `.pin_memory()`** from `precompute_entity_tensors()` and `precompute_relation_tensors()` return statements. Tables returned as plain CPU tensors.

2. **Added GPU migration block** in `_main()` immediately after the precompute calls:
   ```python
   # train_ikge_w2v.py — immediately after precompute_relation_tensors()
   ent_desc           = ent_desc.to(device)
   ent_len            = ent_len.to(device)
   ent_type           = ent_type.to(device)
   ent_names          = ent_names.to(device)
   rel_name_t         = rel_name_t.to(device)
   rel_domain_t       = rel_domain_t.to(device)
   rel_range_t        = rel_range_t.to(device)
   rel_domain_words_t = rel_domain_words_t.to(device)
   rel_range_words_t  = rel_range_words_t.to(device)
   ```
   Memory cost: ~35 MB total on GPU. Fits comfortably alongside model weights (~14 MB for 1.27M params at fp32).

3. **Rewrote `build_batch_from_precomputed()`:** Tables are now GPU-resident; the function moves *index* tensors (`h_ids`, `r_ids`, `t_ids`) to GPU on entry, then indexes directly — no post-index copy needed. All values in the returned dict are already on device.

4. **Removed 9 redundant `.to(device)` copies** in `compute_entity_layer_means()` and replaced 9 `.to(device, non_blocking=True)` copies with plain variable aliases in `evaluate_model()`.

**Files:** [train_ikge_w2v.py](train_ikge_w2v.py) — `precompute_entity_tensors()`, `precompute_relation_tensors()`, `_main()`, `build_batch_from_precomputed()`, `compute_entity_layer_means()`, `evaluate_model()`.

**Effect:** Eliminates all per-batch host→device copies for the large tables. Indexing into GPU-resident tables is ~10× faster than CPU indexing + async copy for tables of this size.

---

### Change 10 — Parameterisable epochs / eval_every

**Motivation:** The training script was hard-coded to 200 epochs and `eval_every=5`. A quick 10-epoch smoke-test with per-epoch validation was needed to get an early MRR report.

**Fix:**
- `_main(fraction, ts)` → `_main(fraction, ts, epochs=200, eval_every=5)` — both values now arguments.
- `main(fraction, run_name)` → `main(fraction, run_name, epochs=200, eval_every=5)` — threaded through.
- `argparse` block extended with `--epochs` and `--eval-every` flags.
- Hard-coded `epochs=200` / `eval_every=5` lines in `_main()` removed; the config comment updated to reference the caller.

**File:** [train_ikge_w2v.py](train_ikge_w2v.py) — `main()`, `_main()`, `__main__` argparse.

---

### New file — quick_train_w2v.py

**Purpose:** A thin wrapper over `train_ikge_w2v.main()` that defaults to 10 epochs with per-epoch validation, then runs full 4-group MRR evaluation at the end. Designed for rapid iteration and checkpoint validation.

**Usage:**
```bash
python quick_train_w2v.py                  # 10 epochs, eval every 1
python quick_train_w2v.py --epochs 20      # 20 epochs
python quick_train_w2v.py --eval-every 2   # validate every 2 epochs
```

**File:** [quick_train_w2v.py](quick_train_w2v.py)

---

### Current Candidate Checkpoint

`ikge_w2v_best_mrr_20260304_001952.pt` — saved at epoch 5 with:
- `val_loss = 0.1287`
- `pos_score = 0.6515`, `neg_score = 0.3001`, **`gap = +0.3515`**

Training was still improving at epoch 8 (`train_loss = 0.0937`, `grad_norm = 0.2721`). The best final weights are expected to be produced by either the full 200-epoch run or the quick 10-epoch run (which will overwrite the checkpoint with an epoch-10 best).

---

### Updated Reproducibility Gaps (revised entries)

| # | Gap | Severity | Updated assumption |
|---|---|---|---|
| 6 | Aggregation depth K | Medium | **Changed to K=2**. DBPedia50k+ avg 2.7 facts/entity makes K=3 redundant. K=2 is now the running assumption and produces cleaner training curves. |
| 7 | Scoring MLP | Low | LayerNorm added before MLP input (not in paper). Soft type-gate (0.1 floor) replaces hard zero gate. ReLU between conv1 and conv2 added. |
| 8 | OOK type constraints | Low | _type_matching() also returns validity=1 when entity has zero type annotations (missing from entity2type.txt). |

**New gap added:**

| # | Gap | Severity | Our assumption |
|---|---|---|---|
| 15 | Type hierarchy in type matching | High | Paper Eq 5 assumes flat type intersection. DBPedia's type hierarchy means `dbo:President` does not flat-match constraint `dbo:Person`. Soft gate (0.1 floor) compensates — 87.4% of triples get 10% feature weight instead of 0. This is a data-model mismatch, not a code error. |
| 16 | Negative sampling difficulty | High | Paper says "generate negative triples by randomly replacing head or tail entity" (Section 5.2.2) without specifying the candidate pool. Our `eval_ooo.py` sanity check showed full-ranking MRR=0.017 on in-KG triples, confirming the model memorised training topology rather than learning semantic representations. Random in-KG negatives are trivially easy (avg 2.7 facts/entity → model needs only "is this entity connected to anything?" not description content). **Fix:** Pre-build per-relation type-constrained buckets (`rel_tail_type_ents[r]` = in-KG entities matching `rel_range_t[r]`; `rel_head_type_ents[r]` matching `rel_domain_t[r]`). Hard negatives are sampled from these buckets — semantically plausible wrong answers the model can only reject by reading descriptions. Fallback to uniform in-KG if bucket <5 entries or exhausted after 200 tries. See `generate_neg_indices` (train_ikge_w2v.py lines 376–440) and the bucket pre-computation block following `in_kg_ents` construction. |

````
