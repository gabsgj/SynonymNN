"""
Dataset loader and vocabulary management for Word Embedding and Transformations.
Loads semantic pairs (synonyms, antonyms, general relations) and sequences from dataset.json.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import torch
from torch.utils.data import Dataset, DataLoader


class Vocabulary:
    """Vocabulary mapping discrete words to integer indices and one-hot vectors."""
    def __init__(self, words: Optional[List[str]] = None):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        if words:
            for w in words:
                self.add_word(w)

    def add_word(self, word: str) -> int:
        word = word.strip().lower()
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            return idx
        return self.word2idx[word]

    def __len__(self) -> int:
        return len(self.word2idx)

    def __contains__(self, word: str) -> bool:
        return word.strip().lower() in self.word2idx

    def get_idx(self, word: str) -> int:
        word = word.strip().lower()
        if word not in self.word2idx:
            raise KeyError(f"Word '{word}' not in vocabulary (size: {len(self.word2idx)})")
        return self.word2idx[word]

    def get_word(self, idx: int) -> str:
        return self.idx2word[idx]

    def to_one_hot(self, word: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """Create a 1-hot row vector x in R^(1 x V) as defined in Section 2."""
        idx = self.get_idx(word)
        x = torch.zeros(1, len(self), device=device)
        x[0, idx] = 1.0
        return x

    def batch_to_one_hot(self, indices: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        """Create batch of one-hot vectors (batch_size, V)."""
        batch_size = indices.size(0)
        x = torch.zeros(batch_size, len(self), device=device or indices.device)
        x.scatter_(1, indices.unsqueeze(1), 1.0)
        return x


class SemanticDataset:
    """
    Parses dataset.json containing:
      - synonyms: list of (word_i, word_j)
      - antonyms: list of (word_i, word_j)
      - relations: dict of relation_name -> list of (word_i, word_j)
      - sequences: list of natural text sentences for sequence modeling
    """
    def __init__(self, json_path: str = "dataset.json"):
        path = Path(json_path)
        if not path.is_absolute():
            path = Path(__file__).parent / json_path
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.raw_data = data
        self.vocab = Vocabulary()

        # Build vocabulary from all words
        self.synonyms: List[Tuple[str, str]] = []
        for pair in data.get("synonyms", []):
            w1, w2 = pair[0].lower(), pair[1].lower()
            self.vocab.add_word(w1)
            self.vocab.add_word(w2)
            self.synonyms.append((w1, w2))
            # Synonyms are symmetric
            self.synonyms.append((w2, w1))

        self.antonyms: List[Tuple[str, str]] = []
        for pair in data.get("antonyms", []):
            w1, w2 = pair[0].lower(), pair[1].lower()
            self.vocab.add_word(w1)
            self.vocab.add_word(w2)
            self.antonyms.append((w1, w2))
            # Antonyms are symmetric
            self.antonyms.append((w2, w1))

        self.relations: Dict[str, List[Tuple[str, str]]] = {}
        for rel_name, pairs in data.get("relations", {}).items():
            rel_pairs = []
            for pair in pairs:
                w1, w2 = pair[0].lower(), pair[1].lower()
                self.vocab.add_word(w1)
                self.vocab.add_word(w2)
                rel_pairs.append((w1, w2))
            self.relations[rel_name] = rel_pairs

        self.sequences: List[List[str]] = []
        for seq_str in data.get("sequences", []):
            words = [w.strip().lower() for w in seq_str.split() if w.strip()]
            for w in words:
                self.vocab.add_word(w)
            if len(words) >= 2:
                self.sequences.append(words)

        # De-duplicate
        self.synonyms = list(set(self.synonyms))
        self.antonyms = list(set(self.antonyms))

    @property
    def relation_names(self) -> List[str]:
        return ["synonym", "antonym"] + list(self.relations.keys())

    def get_semantic_pairs(self) -> List[Tuple[int, int, str]]:
        """Returns (input_idx, target_idx, relation_name) tuples."""
        examples = []
        for w1, w2 in self.synonyms:
            examples.append((self.vocab.get_idx(w1), self.vocab.get_idx(w2), "synonym"))
        for w1, w2 in self.antonyms:
            examples.append((self.vocab.get_idx(w1), self.vocab.get_idx(w2), "antonym"))
        for rel_name, pairs in self.relations.items():
            for w1, w2 in pairs:
                examples.append((self.vocab.get_idx(w1), self.vocab.get_idx(w2), rel_name))
        return examples

    def get_sequence_examples(self) -> List[Tuple[List[int], int]]:
        """
        Extracts multiword-to-next-word training pairs:
        Given prefix [w1, ..., wt], target is wt+1.
        e.g., ["the", "cat", "drinks"] -> "milk"
        """
        seq_examples = []
        for seq in self.sequences:
            indices = [self.vocab.get_idx(w) for w in seq]
            for t in range(1, len(indices)):
                prefix = indices[:t]
                target = indices[t]
                seq_examples.append((prefix, target))
        return seq_examples

    def summary(self) -> str:
        s = [
            f"--- Semantic Dataset Summary ---",
            f"Vocabulary Size (V): {len(self.vocab)}",
            f"Synonym Pairs:       {len(self.synonyms)}",
            f"Antonym Pairs:       {len(self.antonyms)}",
            f"Relations:           {list(self.relations.keys())}",
            f"Relation Pairs:      {sum(len(p) for p in self.relations.values())}",
            f"Sequences (Sentences): {len(self.sequences)}",
            f"Sequence Sub-examples: {len(self.get_sequence_examples())}",
        ]
        return "\n".join(s)


if __name__ == "__main__":
    ds = SemanticDataset("dataset.json")
    print(ds.summary())
