# Auto-generated from Notebooks/FeatureEngineering.ipynb
# Contains code cells only (markdown removed)

# GPU check for future pipeline
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)

if not torch.cuda.is_available():
  print("------------No GPU. Set Runtime → Change runtime type → GPU------------")

try:
    import torch_geometric
    print("Torch Geometric:", torch_geometric.__version__)
except ModuleNotFoundError:
    print("Torch Geometric not found. Installing")
    torch_version = torch.__version__.split("+")[0]
    cuda_version = torch.version.cuda.replace(".", "")

    !python -m pip install -q pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
        -f https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_version}.html

    !python -m pip install -q torch-geometric




# =========================
# Dataset download & normalization
# =========================

from pathlib import Path
import requests
import pandas as pd
from torch_geometric.datasets import WordNet18RR, FB15k_237

# -------------------------
# Paths
# -------------------------
RAW_DIR  = Path("./raw_data")      # Raw / potentially dirty datasets
DATA_DIR = Path("./data/newlinks") # Normalized datasets (h, r, t) for New Links
OOKB_DIR = Path("./data/newentities") # Normalized datasets (h, r, t) for New Entities

OOKB_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(parents=True,exist_ok=True)

print(f"RAW_DIR : {RAW_DIR.resolve()}")
print(f"DATA_DIR: {DATA_DIR.resolve()}")

# -------------------------
# Helpers
# -------------------------
def normalize_to_txt(src_path: Path, dst_path: Path):
    """
    Read raw KG triple file src_path and saves first 3 columns
    as head<TAB>rel<TAB>tail into dst_path.
    """
    df = pd.read_csv(
        src_path,
        sep=None,
        engine="python",
        header=None,
        on_bad_lines="skip"
    )

    if df.shape[1] < 3:
        raise ValueError(
            f"[FORMAT ERROR] Invalid KG triple file: {src_path}\n"
            f"Detected columns: {df.shape[1]}\n"
            "Expected format: head, relation, tail, [optional extra columns]"
        )

    df.iloc[:, :3].to_csv(dst_path, sep="\t", index=False, header=False)


def pyg_dataset_to_standard(pyg_dataset, name: str):
    """
    Normalize (tab) PyG raw files from raw
    and saves as data/name/{train,valid,test}.txt
    into data/name
    """
    raw_dir = Path(pyg_dataset.raw_dir)
    out_dir = DATA_DIR / name
    out_dir.mkdir(exist_ok=True)

    print(f"\nProcessing PyG dataset: {name}")

    file_map = {
        "train": ["train.txt"],
        "valid": ["valid.txt", "valid.csv"],
        "test":  ["test.txt"]
    }

    for split, candidates in file_map.items():
        for fname in candidates:
            src = raw_dir / fname
            if src.exists():
                dst = out_dir / f"{split}.txt"
                normalize_to_txt(src, dst)
                print(f"  -> {split}.txt")
                break
        else:
            print(f"  [!] Missing split: {split}")


def download_file(url: str, dst: Path):
    if dst.exists():
        return
    print(f"Downloading {dst.name}...")
    r = requests.get(url)
    r.raise_for_status()
    dst.write_bytes(r.content)

# -------------------------
# PyG datasets
# -------------------------
print("\n--- Downloading PyG datasets ---")

wn18rr = WordNet18RR(root=RAW_DIR / "WordNet18RR")
pyg_dataset_to_standard(wn18rr, "WN18RR")

fb237 = FB15k_237(root=RAW_DIR / "FB15k-237")
pyg_dataset_to_standard(fb237, "FB15k-237")

# -------------------------
# External datasets
# -------------------------
print("\n--- Downloading external datasets ---")

EXTERNAL_DATASETS = {
    "CoDEx-M": "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-m/",
    "WN11":    "https://raw.githubusercontent.com/KGCompletion/TransL/master/WN11/",
    "FB13":    "https://raw.githubusercontent.com/KGCompletion/TransL/master/FB13/",
}

for name, base_url in EXTERNAL_DATASETS.items():
    raw_out = RAW_DIR / name
    data_out = DATA_DIR / name
    raw_out.mkdir(exist_ok=True)
    data_out.mkdir(exist_ok=True)

    print(f"\n{name}")
    for split in ["train", "valid", "test"]:
        url = f"{base_url}{split}.txt"
        raw_path = raw_out / f"{split}.txt"
        data_path = data_out / f"{split}.txt"

        download_file(url, raw_path)
        normalize_to_txt(raw_path, data_path)
        print(f"  -> {split}.txt")

    if name != "CoDEx-M":
      for split in ["entity2id", "relation2id"]:
          url = f"{base_url}{split}.txt"
          raw_path = raw_out / f"{split}.txt"
          download_file(url, raw_path)

print("\n[DONE] All datasets downloaded and normalized.")


# =========================
# Inductive relation-based splits (NL-*)
# =========================

from pathlib import Path
import random
from collections import defaultdict

# -------------------------
# Config
# -------------------------
SEED = 42

# Defined as in paper, scenarios for different persentages of
# dataset links used as unseen in training
ALPHAS = {
    "NL-25": 0.25,
    "NL-50": 0.50,
    "NL-75": 0.75,
    "NL-100": 1.00,
}

# Reproducibility
random.seed(SEED)

# -------------------------
# IO helpers
# -------------------------
def read_triples(path: Path):
    triples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            h, r, t = line.rstrip("\n").split("\t")
            triples.append((h, r, t))
    return triples


def write_triples(path: Path, triples):
    with path.open("w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


# -------------------------
# Core logic
# -------------------------
def generate_inductive_splits(dataset_dir: Path):
    """
    Generate inductive relation-based splits (NL-*) for a dataset directory.

    The input directory must contain:
        train.txt
        valid.txt
        test.txt

    The function creates, inside the same directory:
        NL-25/, NL-50/, NL-75/, NL-100/
    each containing train/valid/test splits where relations in valid/test
    are completely unseen during training.

    Parameters
    ----------
    dataset_dir : Path
        Path to a dataset directory under BASE_DATA_DIR.
    """
    train_path = dataset_dir / "train.txt"
    valid_path = dataset_dir / "valid.txt"
    test_path  = dataset_dir / "test.txt"

    if not (train_path.exists() and valid_path.exists() and test_path.exists()):
        print(f"[SKIP] {dataset_dir.name}: missing train/valid/test files")
        return

    print(f"\n[DATASET] {dataset_dir.name}")

    train = read_triples(train_path)
    valid = read_triples(valid_path)
    test  = read_triples(test_path)

    # All triples of dataset
    all_triples = train + valid + test

    # Group triples by relation
    rel2triples = defaultdict(list)
    for h, r, t in all_triples:
        rel2triples[r].append((h, r, t))

    # Relations
    relations = list(rel2triples.keys())
    num_relations = len(relations)

    print(f"  Total relations : {num_relations}")
    print(f"  Total triples   : {len(all_triples)}")

    for split_name, alpha in ALPHAS.items():
        # New = unseen at training triples
        # number of new triples
        n_new = int(round(num_relations * alpha))

        # Randomly selected
        shuffled = relations[:]
        random.shuffle(shuffled)

        # new links -> val/test
        # old links -> train
        new_rels = set(shuffled[:n_new])
        old_rels = set(shuffled[n_new:])

        # old links -> train
        train_split = []
        for r in old_rels:
            train_split.extend(rel2triples[r])

        # new links -> val/test
        new_triples = []
        for r in new_rels:
            new_triples.extend(rel2triples[r])

        # val/test -> 50%/50% of new links total
        random.shuffle(new_triples)
        mid = len(new_triples) // 2
        valid_split = new_triples[:mid]
        test_split  = new_triples[mid:]

        # Safety checks
        assert {r for _, r, _ in train_split}.isdisjoint(new_rels)
        assert {r for _, r, _ in valid_split}.issubset(new_rels)
        assert {r for _, r, _ in test_split}.issubset(new_rels)

        out_dir = dataset_dir / split_name
        out_dir.mkdir(exist_ok=True)

        write_triples(out_dir / "train.txt", train_split)
        write_triples(out_dir / "valid.txt", valid_split)
        write_triples(out_dir / "test.txt",  test_split)

        print(
            f"  [{split_name}] "
            f"new_rel={len(new_rels)} | "
            f"train={len(train_split)} | "
            f"valid={len(valid_split)} | "
            f"test={len(test_split)}"
        )


# -------------------------
# Run for all datasets
# -------------------------
print("\n=== Generating inductive splits for all datasets ===")

for dataset_dir in DATA_DIR.iterdir():
    if dataset_dir.is_dir():
        generate_inductive_splits(dataset_dir)

print("\n[DONE] All NL-* splits generated.")


# -------------------------
# Config
# -------------------------
UNSEEN_RATIO = 0.20   # % de entidades OOKB

# -------------------------
# OOKB logic
# -------------------------
def generate_ookb_splits(dataset_dir: Path):
    train_path = dataset_dir / "train.txt"
    valid_path = dataset_dir / "valid.txt"
    test_path  = dataset_dir / "test.txt"

    if not (train_path.exists() and valid_path.exists() and test_path.exists()):
        print(f"[SKIP] {dataset_dir.name}: missing train/valid/test")
        return

    print(f"\n[OOKB DATASET] {dataset_dir.name}")

    train = read_triples(train_path)
    valid = read_triples(valid_path)
    test  = read_triples(test_path)

    # All triples of dataset
    all_triples = train + valid + test

    # Collect entities & relations
    entities = set()
    relations = set()
    for h, r, t in all_triples:
        entities.update([h, t])
        relations.add(r)

    entities  = list(entities)
    relations = list(relations)

    # Select unseen entities
    random.shuffle(entities)
    # Ratio-based selection
    n_unseen = int(round(len(entities) * UNSEEN_RATIO))
    unseen_entities = set(entities[:n_unseen])

    print(f"  entities={len(entities)} | unseen={len(unseen_entities)}")

    # Split triples
    train_split = []
    new_triples = []

    # new triples = triples not seen at training
    # train set directly assigned
    for h, r, t in all_triples:
        if h in unseen_entities or t in unseen_entities:
            new_triples.append((h, r, t))
        else:
            train_split.append((h, r, t))

    random.shuffle(new_triples)
    mid = len(new_triples) // 2

    # val/test -> 50%/50% of new links total
    valid_split = new_triples[:mid]
    test_split  = new_triples[mid:]

    # -------- safety checks --------
    assert all(
        h not in unseen_entities and t not in unseen_entities
        for h, _, t in train_split
    )

    assert any(
        h in unseen_entities or t in unseen_entities
        for h, _, t in valid_split + test_split
    )

    # -------- output --------
    out_dir = OOKB_DIR / dataset_dir.name
    out_dir.mkdir(exist_ok=True)

    write_triples(out_dir / "train.txt", train_split)
    write_triples(out_dir / "valid.txt", valid_split)
    write_triples(out_dir / "test.txt",  test_split)

    # In order to replicate OOKB structure
    # dictionaries need to be generated
    # -------- dictionaries --------
    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}
    unseenentity2id = {e: entity2id[e] for e in sorted(unseen_entities)}

    with (out_dir / "entity2id.txt").open("w") as f:
        for e, i in entity2id.items():
            f.write(f"{e}\t{i}\n")

    with (out_dir / "relation2id.txt").open("w") as f:
        for r, i in relation2id.items():
            f.write(f"{r}\t{i}\n")

    # Unseen dict is intended to be used as a reference
    # to perform semantic or text verification after predictions
    with (out_dir / "unseenentity2id.txt").open("w") as f:
        for e, i in unseenentity2id.items():
            f.write(f"{e}\t{i}\n")

    print(
        f"  train={len(train_split)} | "
        f"valid={len(valid_split)} | "
        f"test={len(test_split)}"
    )


# -------------------------
# Run OOKB for all datasets
# -------------------------

import shutil

OOKB_PREDEFINED = {"WN11", "FB13"}

# Predefined datasets already meet OOKB requirements
print("\n=== Preparing PREDEFINED OOKB datasets (from RAW) ===")

for name in OOKB_PREDEFINED:
    src = RAW_DIR / name
    dst = OOKB_DIR / name

    if not src.exists():
        print(f"[SKIP] {name}: not found in RAW_DIR")
        continue

    if dst.exists():
        print(f"[OK] {name}: already exists")
        continue

    shutil.copytree(src, dst)
    print(f"[COPIED] {name}")


# Custom datasets need to be prepared explicitly
print("\n=== Generating OOKB splits (custom datasets only) ===")

for dataset_dir in DATA_DIR.iterdir():
    if not dataset_dir.is_dir():
        continue

    if dataset_dir.name in OOKB_PREDEFINED:
        continue   # WN11 / FB13 ya están listos

    generate_ookb_splits(dataset_dir)

print("\n[DONE] All OOKB datasets generated.")
