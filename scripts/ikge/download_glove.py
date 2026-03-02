"""
Download and Prepare GloVe Word Embeddings
==========================================

GloVe (Global Vectors for Word Representation) provides pre-trained word embeddings.
IKGE uses these to initialize word representations for entity descriptions.

Paper uses 300-dimensional word embeddings (GloVe or Word2Vec)
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_glove(output_dir='./embeddings', version='6B', dimension=300):
    """
    Download GloVe pre-trained embeddings.
    
    Args:
        output_dir: Where to save embeddings
        version: GloVe version ('6B', '42B', '840B', 'twitter.27B')
        dimension: Embedding dimension (50, 100, 200, 300)
    
    Available versions:
        - '6B': 6 billion tokens, 400K vocab (Wikipedia 2014 + Gigaword 5)
        - '42B': 42 billion tokens, 1.9M vocab (Common Crawl)
        - '840B': 840 billion tokens, 2.2M vocab (Common Crawl)
        - 'twitter.27B': 27 billion tokens, 1.2M vocab (Twitter)
    
    Recommended: '6B' with 300d (good balance of quality and size)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # File URLs
    urls = {
        '6B': 'https://nlp.stanford.edu/data/glove.6B.zip',
        '42B': 'https://nlp.stanford.edu/data/glove.42B.300d.zip',
        '840B': 'https://nlp.stanford.edu/data/glove.840B.300d.zip',
        'twitter.27B': 'https://nlp.stanford.edu/data/glove.twitter.27B.zip'
    }
    
    if version not in urls:
        raise ValueError(f"Invalid version. Choose from: {list(urls.keys())}")
    
    # Check if already downloaded
    if version == '6B':
        target_file = output_dir / f'glove.6B.{dimension}d.txt'
    else:
        target_file = output_dir / f'glove.{version}.300d.txt'
    
    if target_file.exists():
        print(f"✅ GloVe embeddings already exist: {target_file}")
        print(f"   Size: {target_file.stat().st_size / (1024**2):.1f} MB")
        return str(target_file)
    
    # Download
    url = urls[version]
    zip_path = output_dir / f'glove_{version}.zip'
    
    if zip_path.exists():
        print(f"⏭️  Zip already downloaded: {zip_path} ({zip_path.stat().st_size / (1024**2):.1f} MB) — skipping download")
    else:
        print(f"📥 Downloading GloVe {version} embeddings...")
        print(f"   URL: {url}")
        print(f"   Destination: {zip_path}")
        print(f"   This may take a while (file is large)...")
        
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc='Downloading') as t:
            urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)
        
        print(f"\n✅ Download complete: {zip_path.stat().st_size / (1024**2):.1f} MB")
    
    # Extract
    print(f"\n📦 Extracting embeddings to {output_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    # Remove zip file to save space
    zip_path.unlink()
    print(f"🗑️  Removed zip file to save space")
    
    print(f"\n✅ GloVe embeddings ready: {target_file}")
    return str(target_file)


def load_glove_embeddings(glove_file, vocab=None, embedding_dim=300):
    """
    Load GloVe embeddings into a dictionary.
    
    Args:
        glove_file: Path to GloVe .txt file
        vocab: Optional vocabulary set (only load words in vocab)
        embedding_dim: Embedding dimension
        
    Returns:
        embeddings_dict: Dictionary mapping word -> embedding vector
        embedding_matrix: Numpy array for unknown words
    """
    print(f"\n📖 Loading GloVe embeddings from: {glove_file}")
    
    embeddings_dict = {}
    
    with open(glove_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        for line in tqdm(lines, desc="Loading embeddings"):
            values = line.split()
            word = values[0]
            
            # Skip if not in vocabulary (faster loading)
            if vocab is not None and word not in vocab:
                continue
            
            try:
                vector = np.array(values[1:], dtype='float32')
                if len(vector) == embedding_dim:
                    embeddings_dict[word] = vector
            except:
                continue
    
    print(f"✅ Loaded {len(embeddings_dict):,} word embeddings")
    
    # Create default embedding for unknown words (mean of all embeddings)
    if embeddings_dict:
        all_embeddings = np.array(list(embeddings_dict.values()))
        mean_embedding = np.mean(all_embeddings, axis=0)
    else:
        mean_embedding = np.zeros(embedding_dim, dtype='float32')
    
    return embeddings_dict, mean_embedding


def create_embedding_matrix(word2idx, glove_dict, mean_embedding, embedding_dim=300):
    """
    Create embedding matrix for PyTorch nn.Embedding layer.
    
    Args:
        word2idx: Dictionary mapping word -> index
        glove_dict: Dictionary mapping word -> GloVe vector
        mean_embedding: Default embedding for unknown words
        embedding_dim: Embedding dimension
        
    Returns:
        embedding_matrix: Tensor of shape (vocab_size, embedding_dim)
        coverage: Percentage of vocabulary covered by GloVe
    """
    vocab_size = len(word2idx)
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype='float32')
    
    found = 0
    for word, idx in word2idx.items():
        if word in glove_dict:
            embedding_matrix[idx] = glove_dict[word]
            found += 1
        else:
            # Paper: Kaiming uniform (He) initialisation for OOV words.
            # fan_in = embedding_dim; bound = sqrt(1/fan_in) * sqrt(3)
            bound = (1.0 / np.sqrt(embedding_dim)) * np.sqrt(3.0)
            embedding_matrix[idx] = np.random.uniform(-bound, bound, embedding_dim).astype('float32')

    coverage = (found / vocab_size) * 100

    print(f"\n📊 Embedding Matrix Statistics:")
    print(f"   Vocabulary size: {vocab_size:,}")
    print(f"   Words found in GloVe: {found:,} ({coverage:.1f}%)")
    print(f"   Words using Kaiming uniform init: {vocab_size - found:,}")
    
    return torch.FloatTensor(embedding_matrix), coverage


def build_vocabulary_from_descriptions(entity_descriptions, min_freq=1):
    """
    Build vocabulary from entity descriptions.
    
    Args:
        entity_descriptions: List of description strings
        min_freq: Minimum frequency for a word to be included
        
    Returns:
        word2idx: Dictionary mapping word -> index
        idx2word: Dictionary mapping index -> word
        word_freq: Dictionary mapping word -> frequency
    """
    print(f"\n📚 Building vocabulary from {len(entity_descriptions)} descriptions...")
    
    from collections import Counter
    
    # Tokenize and count words
    word_freq = Counter()
    
    for desc in tqdm(entity_descriptions, desc="Tokenizing"):
        if isinstance(desc, str):
            # Simple tokenization (split on whitespace and lowercase)
            words = desc.lower().split()
            word_freq.update(words)
    
    # Filter by frequency
    vocab_words = [word for word, freq in word_freq.items() if freq >= min_freq]
    vocab_words = sorted(vocab_words)  # Alphabetical order
    
    # Create mappings
    word2idx = {word: idx + 2 for idx, word in enumerate(vocab_words)}  # Start at 2
    word2idx['<PAD>'] = 0  # Padding token
    word2idx['<UNK>'] = 1  # Unknown token
    
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    print(f"✅ Vocabulary built:")
    print(f"   Total unique words: {len(word_freq):,}")
    print(f"   Words with freq >= {min_freq}: {len(vocab_words):,}")
    print(f"   Final vocabulary size: {len(word2idx):,}")
    
    return word2idx, idx2word, word_freq


# ============================================================================
# Complete Setup Pipeline
# ============================================================================

def setup_glove_for_ikge(entity_descriptions, output_dir='./embeddings', 
                         glove_version='6B', embedding_dim=300):
    """
    Complete pipeline to set up GloVe embeddings for IKGE.
    
    Args:
        entity_descriptions: List of entity description strings
        output_dir: Where to save embeddings
        glove_version: GloVe version to download
        embedding_dim: Embedding dimension
        
    Returns:
        embedding_matrix: PyTorch tensor ready for nn.Embedding
        word2idx: Word to index mapping
        idx2word: Index to word mapping
    """
    print("=" * 80)
    print("🚀 SETTING UP GLOVE EMBEDDINGS FOR IKGE")
    print("=" * 80)
    
    # Step 1: Download GloVe
    glove_file = download_glove(output_dir, glove_version, embedding_dim)
    
    # Step 2: Build vocabulary from descriptions
    word2idx, idx2word, word_freq = build_vocabulary_from_descriptions(entity_descriptions)
    
    # Step 3: Load GloVe (only words in our vocabulary for efficiency)
    vocab_set = set(word2idx.keys())
    glove_dict, mean_embedding = load_glove_embeddings(glove_file, vocab_set, embedding_dim)
    
    # Step 4: Create embedding matrix
    embedding_matrix, coverage = create_embedding_matrix(
        word2idx, glove_dict, mean_embedding, embedding_dim
    )
    
    print("\n" + "=" * 80)
    print("✅ GLOVE SETUP COMPLETE!")
    print("=" * 80)
    print(f"\n📦 Ready to use:")
    print(f"   - embedding_matrix: shape {embedding_matrix.shape}")
    print(f"   - word2idx: {len(word2idx):,} words")
    print(f"   - GloVe coverage: {coverage:.1f}%")
    
    return embedding_matrix, word2idx, idx2word


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import pandas as pd
    
    print("\n" + "=" * 80)
    print("🧪 TESTING GLOVE DOWNLOAD AND SETUP")
    print("=" * 80)
    
    # Test with example descriptions
    test_descriptions = [
        "Boston is the capital and most populous city of Massachusetts",
        "Harvard University is a private Ivy League research university",
        "Massachusetts is a U.S. state in the New England region",
    ]
    
    # Run setup
    embedding_matrix, word2idx, idx2word = setup_glove_for_ikge(
        entity_descriptions=test_descriptions,
        output_dir='./embeddings',
        glove_version='6B',
        embedding_dim=300
    )
    
    # Test embedding lookup
    print("\n🔍 Testing word embeddings:")
    test_words = ['boston', 'university', 'unknown_word_xyz']
    
    for word in test_words:
        if word in word2idx:
            idx = word2idx[word]
            embedding = embedding_matrix[idx]
            print(f"   '{word}' -> index {idx}, embedding shape: {embedding.shape}")
        else:
            print(f"   '{word}' -> NOT IN VOCABULARY (will use <UNK>)")
    
    print("\n" + "=" * 80)
    print("💡 INTEGRATION WITH IKGE:")
    print("=" * 80)
    print("""
# In your data loader:
entity_df = pd.read_csv('data/codex-m/enriched/entity_descriptions.csv')
descriptions = entity_df['description'].fillna('').tolist()

# Setup GloVe
embedding_matrix, word2idx, idx2word = setup_glove_for_ikge(
    entity_descriptions=descriptions,
    output_dir='./embeddings',
    glove_version='6B',
    embedding_dim=300
)

# In your IKGE model:
class FactFeatureExtractor(nn.Module):
    def __init__(self, embedding_matrix, ...):
        # Initialize with pre-trained GloVe
        self.word_embeddings = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True  # Don't train word embeddings
        )
        ...
""")