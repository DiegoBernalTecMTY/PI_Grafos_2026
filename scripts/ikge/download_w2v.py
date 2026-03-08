"""
Download and Prepare Wikipedia2Vec Word Embeddings
===================================================

Wikipedia2Vec provides pre-trained word embeddings trained directly on Wikipedia.
This gives ~100% coverage for DBPedia entity descriptions (which are sourced from
Wikipedia), versus ~29% coverage with GloVe 6B.

Paper (Hwang et al., Information Sciences 2022) uses Wikipedia2Vec 300-dim.

Same public interface as download_glove.py:
    embedding_matrix, word2idx, idx2word = setup_w2v_for_ikge(
        entity_descriptions, output_dir, embedding_dim
    )
"""

import os
import re
import bz2
import urllib.request
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared tokenizer  — must be used identically during vocab build AND inference
# Strips punctuation from word boundaries so tokens like "city," and "born."
# match Wikipedia2Vec's clean-word vocabulary.
# ---------------------------------------------------------------------------
_WORD_SPLIT = re.compile(r"[^a-z0-9']+")

# ---------------------------------------------------------------------------
# Lemmatizer  (paper Section 5.1.1: "we perform lemmatization to extract the
# basic forms of the words")
# ---------------------------------------------------------------------------
try:
    from nltk.stem import WordNetLemmatizer as _WNL
    import nltk as _nltk
    _lemmatizer = _WNL()
    # Trigger a quick test; download wordnet data if absent.
    try:
        _lemmatizer.lemmatize("running")
    except LookupError:
        _nltk.download("wordnet", quiet=True)
        _nltk.download("omw-1.4", quiet=True)
        _lemmatizer.lemmatize("running")  # confirm it works now
    def _lemmatize(word: str) -> str:
        return _lemmatizer.lemmatize(word)
except ImportError:
    import warnings as _warnings
    _warnings.warn(
        "nltk not installed — lemmatization disabled. "
        "Install with: pip install nltk  (paper Section 5.1.1 requires lemmatization).",
        RuntimeWarning,
        stacklevel=2,
    )
    def _lemmatize(word: str) -> str:  # no-op fallback
        return word


def tokenize_for_w2v(text: str) -> list:
    """Lowercase + strip punctuation + lemmatize (paper Section 5.1.1)."""
    return [_lemmatize(w) for w in _WORD_SPLIT.split(text.lower()) if w]


# ---------------------------------------------------------------------------
# Pretrained model URLs  (source: wikipedia2vec.github.io/pretrained/)
# ---------------------------------------------------------------------------
W2V_URLS = {
    100: "http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.pkl.bz2",
    300: "http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_300d.pkl.bz2",
}


class DownloadProgressBar(tqdm):
    """Progress bar for urllib.request.urlretrieve."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_w2v(output_dir: str = "./embeddings", embedding_dim: int = 300) -> str:
    """
    Download the Wikipedia2Vec pretrained model (.pkl) for the given dimension.

    Returns the path to the unpacked .pkl file.
    """
    if embedding_dim not in W2V_URLS:
        raise ValueError(f"No Wikipedia2Vec model for dim={embedding_dim}. "
                         f"Available: {list(W2V_URLS)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_name  = f"enwiki_20180420_{embedding_dim}d.pkl"
    pkl_path  = output_dir / pkl_name
    bz2_path  = output_dir / (pkl_name + ".bz2")

    if pkl_path.exists():
        size_mb = pkl_path.stat().st_size / (1024 ** 2)
        print(f"✅ Wikipedia2Vec model already exists: {pkl_path}  ({size_mb:.1f} MB)")
        return str(pkl_path)

    url = W2V_URLS[embedding_dim]

    if not bz2_path.exists():
        print(f"📥 Downloading Wikipedia2Vec {embedding_dim}d model…")
        print(f"   URL : {url}")
        print(f"   Dest: {bz2_path}")
        print("   This file is large (~2-3 GB). Please be patient.")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1,
                                 desc='Downloading') as t:
            urllib.request.urlretrieve(url, bz2_path, reporthook=t.update_to)
        print(f"\n✅ Download complete: {bz2_path.stat().st_size / (1024**2):.1f} MB")
    else:
        print(f"⏭️  Found existing archive {bz2_path} — skipping download.")

    # Decompress
    print(f"📦 Decompressing {bz2_path} → {pkl_path} …")
    with bz2.open(str(bz2_path), 'rb') as src, open(pkl_path, 'wb') as dst:
        chunk_size = 4 * 1024 * 1024  # 4 MB chunks
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break
            dst.write(chunk)

    # Remove archive to save space
    bz2_path.unlink()
    print(f"🗑️  Removed archive to save space.")
    print(f"✅ Wikipedia2Vec model ready: {pkl_path}  "
          f"({pkl_path.stat().st_size / (1024**2):.1f} MB)")
    return str(pkl_path)


# ---------------------------------------------------------------------------
# Vocabulary builder  (identical logic to download_glove.py)
# ---------------------------------------------------------------------------

def build_vocabulary_from_descriptions(entity_descriptions, min_freq: int = 1):
    """
    Build word2idx / idx2word from a list of description strings.

    Returns:
        word2idx : dict  word → int index (0-based; 0=<PAD>, 1=<UNK>)
        idx2word : dict  int index → word
        word_freq: Counter
    """
    print(f"\n📚 Building vocabulary from {len(entity_descriptions):,} descriptions…")

    word_freq = Counter()
    for desc in tqdm(entity_descriptions, desc="Tokenising"):
        if isinstance(desc, str):
            word_freq.update(tokenize_for_w2v(desc))

    vocab_words = sorted(w for w, f in word_freq.items() if f >= min_freq)

    word2idx = {w: i + 2 for i, w in enumerate(vocab_words)}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    idx2word = {i: w for w, i in word2idx.items()}

    print(f"✅ Vocabulary built:")
    print(f"   Unique words   : {len(word_freq):,}")
    print(f"   Vocab size     : {len(word2idx):,}")
    return word2idx, idx2word, word_freq


# ---------------------------------------------------------------------------
# Embedding matrix builder
# ---------------------------------------------------------------------------

def create_embedding_matrix_w2v(word2idx: dict, wiki_model,
                                 embedding_dim: int = 300):
    """
    Build a (vocab_size × embedding_dim) matrix using the loaded wikipedia2vec model.

    Words absent from the model get Kaiming-uniform random vectors (same fallback
    as the GloVe pipeline).

    Args:
        word2idx    : vocabulary mapping  word → index
        wiki_model  : loaded Wikipedia2Vec instance
        embedding_dim: dimensionality (must match the loaded model)

    Returns:
        embedding_matrix : torch.FloatTensor  shape (vocab_size, embedding_dim)
        coverage         : float  percentage of vocab covered
    """
    vocab_size       = len(word2idx)
    matrix           = np.zeros((vocab_size, embedding_dim), dtype='float32')
    bound            = (1.0 / np.sqrt(embedding_dim)) * np.sqrt(3.0)  # Kaiming uniform
    found            = 0

    for word, idx in tqdm(word2idx.items(), desc="Building embedding matrix",
                          total=vocab_size):
        entry = wiki_model.get_word(word)
        if entry is not None:
            matrix[idx] = wiki_model.get_word_vector(word).astype('float32')
            found += 1
        else:
            matrix[idx] = np.random.uniform(-bound, bound, embedding_dim).astype('float32')

    coverage = (found / vocab_size) * 100

    print(f"\n📊 Embedding Matrix Statistics:")
    print(f"   Vocabulary size          : {vocab_size:,}")
    print(f"   Words found in W2V model : {found:,} ({coverage:.1f}%)")
    print(f"   Words using Kaiming init : {vocab_size - found:,}")

    return torch.FloatTensor(matrix), coverage


# ---------------------------------------------------------------------------
# Complete setup pipeline  (drop-in replacement for setup_glove_for_ikge)
# ---------------------------------------------------------------------------

def setup_w2v_for_ikge(entity_descriptions, output_dir: str = "./embeddings",
                       embedding_dim: int = 300):
    """
    Complete pipeline to set up Wikipedia2Vec embeddings for IKGE.

    Identical return signature to setup_glove_for_ikge():
        embedding_matrix : torch.FloatTensor  (vocab_size, embedding_dim)
        word2idx         : dict  word → int
        idx2word         : dict  int → word

    Args:
        entity_descriptions : list of entity / relation description strings
        output_dir          : directory to cache the downloaded model
        embedding_dim       : 100 or 300 (must match a pretrained model)
    """
    try:
        from wikipedia2vec import Wikipedia2Vec
    except ImportError:
        raise ImportError(
            "wikipedia2vec is not installed.\n"
            "Run:  pip install wikipedia2vec"
        )

    print("=" * 80)
    print("🚀 SETTING UP WIKIPEDIA2VEC EMBEDDINGS FOR IKGE")
    print("=" * 80)

    # 1. Download / locate the pretrained model
    pkl_path = download_w2v(output_dir, embedding_dim)

    # 2. Load the model (can take 30-60 s for the 300-dim model)
    print(f"\n📖 Loading Wikipedia2Vec model from: {pkl_path}")
    print("   (this may take up to 60 s for the 300-dim model…)")
    import warnings
    with warnings.catch_warnings():
        # The old pickle file has non-byte-aligned mmap arrays (joblib < 1.2).
        # We copy everything into contiguous PyTorch float32 tensors immediately
        # after loading, so the alignment issue has no runtime effect here.
        warnings.filterwarnings("ignore", category=UserWarning,
                                message=".*not byte aligned.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning,
                                message=".*align should be passed.*")
        warnings.filterwarnings("ignore", category=Warning,
                                message=".*align should be passed.*")
        wiki_model = Wikipedia2Vec.load(pkl_path)
    print("✅ Wikipedia2Vec model loaded.")

    # 3. Build vocabulary from entity descriptions
    word2idx, idx2word, _ = build_vocabulary_from_descriptions(entity_descriptions)

    # 4. Build embedding matrix
    embedding_matrix, coverage = create_embedding_matrix_w2v(
        word2idx, wiki_model, embedding_dim
    )

    print("\n" + "=" * 80)
    print("✅ WIKIPEDIA2VEC SETUP COMPLETE!")
    print("=" * 80)
    print(f"\n📦 Ready to use:")
    print(f"   - embedding_matrix : shape {embedding_matrix.shape}")
    print(f"   - word2idx         : {len(word2idx):,} words")
    print(f"   - W2V coverage     : {coverage:.1f}%")

    return embedding_matrix, word2idx, idx2word


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_descriptions = [
        "Boston is the capital and most populous city of Massachusetts",
        "Harvard University is a private Ivy League research university",
        "Massachusetts is a U.S. state in the New England region",
    ]

    embedding_matrix, word2idx, idx2word = setup_w2v_for_ikge(
        entity_descriptions=test_descriptions,
        output_dir="./embeddings",
        embedding_dim=300,
    )

    print("\n🔍 Testing word lookups:")
    for word in ["boston", "university", "unknown_xyz_abc"]:
        if word in word2idx:
            idx = word2idx[word]
            print(f"   '{word}' → idx {idx}, vec shape: {embedding_matrix[idx].shape}")
        else:
            print(f"   '{word}' → NOT IN VOCAB")
