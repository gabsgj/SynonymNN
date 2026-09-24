#!/usr/bin/env python3
"""
Simple & Powerful Raw Implementation of:
'Word Embedding and Word-to-Word Transformations'
Built purely with NumPy (no PyTorch, no TensorFlow, zero heavy frameworks).

Usage:
    python raw_main.py --demo
    python raw_main.py --interactive
    python raw_main.py --synonym happy
    python raw_main.py --antonym hot
    python raw_main.py --relation country_to_capital france
    python raw_main.py --predict "the cat drinks"
    python raw_main.py --similarity happy sad
    python raw_main.py --train [--epochs 50]
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from raw_model import RawUnifiedWordModel, cosine_similarity


class RawDataset:
    """Lightweight pure-Python/NumPy dataset loader for dataset.json."""
    def __init__(self, json_path: str = "dataset.json"):
        path = Path(json_path)
        if not path.is_absolute():
            path = Path(__file__).parent / json_path

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

        def add_word(w: str) -> int:
            w = w.strip().lower()
            if w not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[w] = idx
                self.idx2word[idx] = w
                return idx
            return self.word2idx[w]

        self.synonyms: List[Tuple[str, str]] = []
        for pair in data.get("synonyms", []):
            w1, w2 = pair[0].lower(), pair[1].lower()
            add_word(w1)
            add_word(w2)
            self.synonyms.extend([(w1, w2), (w2, w1)])

        self.antonyms: List[Tuple[str, str]] = []
        for pair in data.get("antonyms", []):
            w1, w2 = pair[0].lower(), pair[1].lower()
            add_word(w1)
            add_word(w2)
            self.antonyms.extend([(w1, w2), (w2, w1)])

        self.relations: Dict[str, List[Tuple[str, str]]] = {}
        for r_name, pairs in data.get("relations", {}).items():
            self.relations[r_name] = []
            for pair in pairs:
                w1, w2 = pair[0].lower(), pair[1].lower()
                add_word(w1)
                add_word(w2)
                self.relations[r_name].append((w1, w2))

        self.sequences: List[List[str]] = []
        for seq_str in data.get("sequences", []):
            tokens = [w.strip().lower() for w in seq_str.split() if w.strip()]
            for w in tokens:
                add_word(w)
            if len(tokens) >= 2:
                self.sequences.append(tokens)

        # De-duplicate
        self.synonyms = list(set(self.synonyms))
        self.antonyms = list(set(self.antonyms))

    def __len__(self) -> int:
        return len(self.word2idx)

    @property
    def relation_names(self) -> List[str]:
        return ["synonym", "antonym"] + list(self.relations.keys())

    def get_semantic_examples(self) -> List[Tuple[int, int, str]]:
        examples = []
        for w1, w2 in self.synonyms:
            examples.append((self.word2idx[w1], self.word2idx[w2], "synonym"))
        for w1, w2 in self.antonyms:
            examples.append((self.word2idx[w1], self.word2idx[w2], "antonym"))
        for r_name, pairs in self.relations.items():
            for w1, w2 in pairs:
                examples.append((self.word2idx[w1], self.word2idx[w2], r_name))
        return examples

    def get_sequence_examples(self) -> List[Tuple[List[int], int]]:
        seq_examples = []
        for seq in self.sequences:
            indices = [self.word2idx[w] for w in seq]
            for t in range(1, len(indices)):
                seq_examples.append((indices[:t], indices[t]))
        return seq_examples


def train_raw_model(
    epochs: int = 50,
    embed_dim: int = 24,
    lr: float = 0.04,
    save_path: str = "raw_model_weights.npz",
    verbose: bool = True
) -> Tuple[RawUnifiedWordModel, RawDataset]:
    dataset = RawDataset("dataset.json")
    model = RawUnifiedWordModel(
        vocab_size=len(dataset),
        embed_dim=embed_dim,
        relation_names=dataset.relation_names
    )

    semantic_data = dataset.get_semantic_examples()
    sequence_data = dataset.get_sequence_examples()

    # Pre-build lookup for validation accuracy
    valid_semantic: Dict[Tuple[str, str], Set[str]] = {}
    for w1, w2 in dataset.synonyms:
        valid_semantic.setdefault((w1, "synonym"), set()).add(w2)
    for w1, w2 in dataset.antonyms:
        valid_semantic.setdefault((w1, "antonym"), set()).add(w2)
    for rel_name, pairs in dataset.relations.items():
        for w1, w2 in pairs:
            valid_semantic.setdefault((w1, rel_name), set()).add(w2)

    valid_seq_targets: Dict[Tuple[int, ...], Set[int]] = {}
    for prefix, target in sequence_data:
        valid_seq_targets.setdefault(tuple(prefix), set()).add(target)

    if verbose:
        print("=" * 85)
        print(" TRAINING RAW NUMPY MODEL (Analytical Gradient Descent & BPTT with Adam)")
        print(f" Vocabulary: {len(dataset)} words | Embed Dim: {embed_dim} | Relations: {len(dataset.relation_names)}")
        print("=" * 85)
        print(f"{'Epoch':<8}{'Sem Loss':<12}{'Seq Loss':<12}{'Syn Acc':<10}{'Ant Acc':<10}{'Seq Acc':<10}{'Syn Cos':<10}{'Ant Cos':<10}")
        print("-" * 85)

    batch_size = 32

    # Group sequences by length
    seq_by_len: Dict[int, List[Tuple[List[int], int]]] = {}
    for prefix, target in sequence_data:
        seq_by_len.setdefault(len(prefix), []).append((prefix, target))

    for epoch in range(1, epochs + 1):
        # 1. Train Semantic Word Transformations (batched by relation)
        random.shuffle(semantic_data)
        sem_by_rel: Dict[str, List[Tuple[int, int]]] = {}
        for inp, tgt, rel in semantic_data:
            sem_by_rel.setdefault(rel, []).append((inp, tgt))

        sem_losses = []
        for rel, pairs in sem_by_rel.items():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                inputs = [p[0] for p in batch]
                targets = [p[1] for p in batch]
                loss = model.train_batch_semantic(inputs, targets, rel, lr=lr, lambda_syn=0.4, lambda_ant=0.4)
                sem_losses.append(loss)
        avg_sem_loss = float(np.mean(sem_losses)) if sem_losses else 0.0

        # 2. Train Multiword Sequences via BPTT (batched by length)
        seq_losses = []
        for length, seq_pairs in seq_by_len.items():
            random.shuffle(seq_pairs)
            for i in range(0, len(seq_pairs), batch_size):
                batch = seq_pairs[i:i + batch_size]
                prefixes = [p[0] for p in batch]
                targets = [p[1] for p in batch]
                loss = model.train_batch_sequence(prefixes, targets, lr=lr)
                seq_losses.append(loss)
        avg_seq_loss = float(np.mean(seq_losses)) if seq_losses else 0.0

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            # Evaluate accuracies
            syn_correct = sum(
                1 for w1 in {p[0] for p in dataset.synonyms}
                if dataset.idx2word[model.predict_semantic(dataset.word2idx[w1], "synonym", top_k=1)[0][0]]
                in valid_semantic.get((w1, "synonym"), set())
            )
            syn_acc = syn_correct / max(1, len({p[0] for p in dataset.synonyms}))

            ant_correct = sum(
                1 for w1 in {p[0] for p in dataset.antonyms}
                if dataset.idx2word[model.predict_semantic(dataset.word2idx[w1], "antonym", top_k=1)[0][0]]
                in valid_semantic.get((w1, "antonym"), set())
            )
            ant_acc = ant_correct / max(1, len({p[0] for p in dataset.antonyms}))

            seq_correct = sum(
                1 for prefix_tup, targets in valid_seq_targets.items()
                if model.predict_next(list(prefix_tup), top_k=1)[0][0] in targets
            )
            seq_acc = seq_correct / max(1, len(valid_seq_targets))

            # Cosine similarities
            syn_cos = np.mean([
                cosine_similarity(model.E[dataset.word2idx[w1]], model.E[dataset.word2idx[w2]])
                for w1, w2 in dataset.synonyms[:40]
            ])
            ant_cos = np.mean([
                cosine_similarity(model.E[dataset.word2idx[w1]], model.E[dataset.word2idx[w2]])
                for w1, w2 in dataset.antonyms[:40]
            ])

            if verbose:
                print(
                    f"{epoch:<8}"
                    f"{avg_sem_loss:<12.4f}"
                    f"{avg_seq_loss:<12.4f}"
                    f"{syn_acc*100:<10.1f}%"
                    f"{ant_acc*100:<10.1f}%"
                    f"{seq_acc*100:<10.1f}%"
                    f"{syn_cos:<10.2f}"
                    f"{ant_cos:<10.2f}"
                )

    # Save weights with pure numpy (.npz)
    save_dict = {
        "E": model.E,
        "D": model.D,
        "T_R": model.T_R,
        "b": model.b,
        "word2idx_json": json.dumps(dataset.word2idx),
        "relation_names_json": json.dumps(dataset.relation_names),
    }
    for r, T_mat in model.T.items():
        save_dict[f"T_{r}"] = T_mat

    np.savez(save_path, **save_dict)
    if verbose:
        print(f"\n[+] Raw model weights saved to {save_path}")

    return model, dataset


def load_raw_model(weights_path: str = "raw_model_weights.npz") -> Tuple[RawUnifiedWordModel, RawDataset]:
    if not os.path.exists(weights_path):
        return train_raw_model(epochs=40, save_path=weights_path)

    data = np.load(weights_path)
    word2idx = json.loads(str(data["word2idx_json"]))
    relation_names = json.loads(str(data["relation_names_json"]))

    dataset = RawDataset("dataset.json")

    E = data["E"]
    vocab_size, embed_dim = E.shape

    model = RawUnifiedWordModel(vocab_size=vocab_size, embed_dim=embed_dim, relation_names=relation_names)
    model.E = E
    model.D = data["D"]
    model.T_R = data["T_R"]
    model.b = data["b"]
    for r in relation_names:
        key = f"T_{r}"
        if key in data:
            model.T[r] = data[key]

    return model, dataset


def run_demo(model: RawUnifiedWordModel, dataset: RawDataset):
    """Walkthrough of all sections of the paper using pure NumPy."""
    print("""
=============================================================================
           RAW NUMPY IMPLEMENTATION (ZERO EXTERNAL FRAMEWORKS)
      Word Embedding & Word-to-Word Transformations (Paper Walkthrough)
=============================================================================
""")
    # 1. Architecture details
    print("1. ARCHITECTURE DIMENSIONS (Sections 1, 3, 4, 8, 10)")
    print(f"   Vocabulary Size (V):     {len(dataset)}")
    print(f"   Embedding Dimension (d):  {model.d}")
    print(f"   Encoder Matrix E:        shape {model.E.shape}")
    print(f"   Decoder Matrix D:        shape {model.D.shape}")
    print(f"   Recurrent Operator T_R:  shape {model.T_R.shape}")
    print(f"   Semantic Transitions:    {list(model.T.keys())}\n")

    # 2. Word-to-Synonym Transition
    print("2. WORD-TO-SYNONYM TRANSITION (Section 5: Tsyn ≈ I, ei ≈ ej)")
    for word in ["happy", "cold", "fast", "bright", "brave", "clean"]:
        if word in dataset.word2idx:
            idx = dataset.word2idx[word]
            preds = model.predict_semantic(idx, "synonym", top_k=4)
            preds_str = [f"{dataset.idx2word[i]} ({p*100:.1f}%)" for i, p in preds]
            print(f"   '{word:<7}' -> Synonyms: {', '.join(preds_str)}")

    T_syn = model.T["synonym"]
    diff_syn = np.linalg.norm(T_syn - np.eye(model.d))
    print(f"   [Math Check] ||Tsyn - I||_F = {diff_syn:.4f} | Avg diag entry: {np.mean(np.diag(T_syn)):.4f}\n")

    # 3. Word-to-Antonym Transition
    print("3. WORD-TO-ANTONYM TRANSITION (Section 6: Tant ≈ -I, ei ≈ -ej)")
    for word in ["hot", "happy", "fast", "big", "dark", "strong"]:
        if word in dataset.word2idx:
            idx = dataset.word2idx[word]
            preds = model.predict_semantic(idx, "antonym", top_k=4)
            preds_str = [f"{dataset.idx2word[i]} ({p*100:.1f}%)" for i, p in preds]
            print(f"   '{word:<7}' -> Antonyms: {', '.join(preds_str)}")

    T_ant = model.T["antonym"]
    diff_ant = np.linalg.norm(T_ant + np.eye(model.d))
    print(f"   [Math Check] ||Tant - (-I)||_F = {diff_ant:.4f} | Avg diag entry: {np.mean(np.diag(T_ant)):.4f}\n")

    # 4. General Semantic Transitions
    print("4. GENERAL SEMANTIC TRANSITIONS (Section 7: Tr in R^(d x d))")
    queries = [
        ("france", "country_to_capital"),
        ("japan", "country_to_capital"),
        ("cat", "singular_to_plural"),
        ("walk", "present_to_past"),
        ("drink", "present_to_past"),
        ("animal", "category_to_instance")
    ]
    for w, r in queries:
        if w in dataset.word2idx and r in model.T:
            idx = dataset.word2idx[w]
            preds = model.predict_semantic(idx, r, top_k=3)
            preds_str = [f"{dataset.idx2word[i]} ({p*100:.1f}%)" for i, p in preds]
            print(f"   Relation [{r}]: '{w}' -> {', '.join(preds_str)}")
    print()

    # 5. Multiword Recurrent Prediction (Section 10-12, 15.3, 15.4)
    print("5. MULTIWORD-TO-WORD PREDICTION (Recurrent Memory ht = tanh([h_{t-1}, et] T_R + b))")
    print("   Example from PDF Section 12: 'The cat drinks' -> 'milk'")
    test_prefixes = [
        "the cat drinks",
        "the dog eats",
        "the dog barks",
        "the bird sings",
        "the sun shines",
        "the summer is",
        "the boy reads",
        "the chef cooks"
    ]
    for prefix in test_prefixes:
        words = prefix.lower().split()
        if all(w in dataset.word2idx for w in words):
            indices = [dataset.word2idx[w] for w in words]
            preds = model.predict_next(indices, top_k=3)
            preds_str = [f"{dataset.idx2word[i]} (p={p:.3f})" for i, p in preds]
            print(f"   Prefix: '{prefix:<16}' -> Next Word: {', '.join(preds_str)}")
    print()

    # 6. Geometric Cosine Similarities
    print("6. VECTOR GEOMETRY & COSINE SIMILARITIES")
    sim_pairs = [
        ("happy", "joyful", "Synonyms (e_i ≈ e_j, expect > 0)"),
        ("hot", "cold", "Antonyms (e_i ≈ -e_j, expect < 0)"),
        ("happy", "sad", "Antonyms (e_i ≈ -e_j, expect < 0)"),
        ("bright", "dark", "Antonyms (e_i ≈ -e_j, expect < 0)"),
        ("fast", "quick", "Synonyms (e_i ≈ e_j, expect > 0)"),
        ("cat", "milk", "Unrelated words (expect near 0)"),
    ]
    for w1, w2, label in sim_pairs:
        if w1 in dataset.word2idx and w2 in dataset.word2idx:
            cos = cosine_similarity(model.E[dataset.word2idx[w1]], model.E[dataset.word2idx[w2]])
            print(f"   cos({w1:<7}, {w2:<7}) = {cos:+.4f} | {label}")
    print("=" * 80 + "\n")


def run_interactive(model: RawUnifiedWordModel, dataset: RawDataset):
    """Interactive loop for raw numpy model."""
    print("\n--- RAW NUMPY INTERACTIVE MODE ---")
    while True:
        print("\nOptions:")
        print("  1. Synonym query (Tsyn ≈ I)")
        print("  2. Antonym query (Tant ≈ -I)")
        print("  3. Semantic Relation query (Tr)")
        print("  4. Next-word prediction from sentence prefix (RNN Transition)")
        print("  5. Cosine similarity between two words")
        print("  0. Exit")

        c = input("\nEnter option [0-5]: ").strip()
        if c in ["0", "exit", "quit", "q"]:
            break
        elif c == "1":
            w = input("Enter word: ").strip().lower()
            if w not in dataset.word2idx:
                print(f"'{w}' not in vocabulary.")
                continue
            for i, p in model.predict_semantic(dataset.word2idx[w], "synonym", top_k=5):
                print(f"  - {dataset.idx2word[i]:<15} ({p*100:.1f}%)")
        elif c == "2":
            w = input("Enter word: ").strip().lower()
            if w not in dataset.word2idx:
                print(f"'{w}' not in vocabulary.")
                continue
            for i, p in model.predict_semantic(dataset.word2idx[w], "antonym", top_k=5):
                print(f"  - {dataset.idx2word[i]:<15} ({p*100:.1f}%)")
        elif c == "3":
            print(f"Relations: {list(model.T.keys())}")
            rel = input("Enter relation: ").strip()
            w = input("Enter word: ").strip().lower()
            if rel not in model.T or w not in dataset.word2idx:
                print("Invalid relation or word.")
                continue
            for i, p in model.predict_semantic(dataset.word2idx[w], rel, top_k=5):
                print(f"  - {dataset.idx2word[i]:<15} ({p*100:.1f}%)")
        elif c == "4":
            text = input("Enter sentence prefix (e.g. 'the cat drinks'): ").strip().lower()
            tokens = text.split()
            if not all(t in dataset.word2idx for t in tokens):
                print("Some words are not in vocabulary.")
                continue
            indices = [dataset.word2idx[t] for t in tokens]
            for i, p in model.predict_next(indices, top_k=5):
                print(f"  - {dataset.idx2word[i]:<15} (p={p:.4f})")
        elif c == "5":
            w1 = input("Word 1: ").strip().lower()
            w2 = input("Word 2: ").strip().lower()
            if w1 not in dataset.word2idx or w2 not in dataset.word2idx:
                print("Words must be in vocabulary.")
                continue
            cos = cosine_similarity(model.E[dataset.word2idx[w1]], model.E[dataset.word2idx[w2]])
            print(f"Cosine similarity cos({w1}, {w2}) = {cos:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Raw NumPy Word Embedding & Transformation System")
    parser.add_argument("--demo", action="store_true", help="Run paper demonstration")
    parser.add_argument("--interactive", action="store_true", help="Interactive CLI console")
    parser.add_argument("--train", action="store_true", help="Train raw numpy model from scratch")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--embed_dim", type=int, default=24, help="Embedding dimension d")
    parser.add_argument("--synonym", type=str, help="Query synonyms for word")
    parser.add_argument("--antonym", type=str, help="Query antonyms for word")
    parser.add_argument("--relation", nargs=2, metavar=("RELATION", "WORD"), help="Query relation for word")
    parser.add_argument("--predict", type=str, help="Predict next word for sequence prefix")
    parser.add_argument("--similarity", nargs=2, metavar=("WORD1", "WORD2"), help="Compute cosine similarity")

    args = parser.parse_args()

    if args.train:
        train_raw_model(epochs=args.epochs, embed_dim=args.embed_dim)
        return

    model, dataset = load_raw_model("raw_model_weights.npz")

    if args.demo:
        run_demo(model, dataset)
    elif args.interactive:
        run_interactive(model, dataset)
    elif args.synonym:
        w = args.synonym.lower()
        if w not in dataset.word2idx:
            print(f"'{w}' not in vocabulary.")
            sys.exit(1)
        for i, p in model.predict_semantic(dataset.word2idx[w], "synonym", top_k=5):
            print(f"  - {dataset.idx2word[i]:<15} ({p*100:.1f}%)")
    elif args.antonym:
        w = args.antonym.lower()
        if w not in dataset.word2idx:
            print(f"'{w}' not in vocabulary.")
            sys.exit(1)
        for i, p in model.predict_semantic(dataset.word2idx[w], "antonym", top_k=5):
            print(f"  - {dataset.idx2word[i]:<15} ({p*100:.1f}%)")
    elif args.relation:
        rel, w = args.relation[0], args.relation[1].lower()
        if rel not in model.T or w not in dataset.word2idx:
            print("Invalid relation or word.")
            sys.exit(1)
        for i, p in model.predict_semantic(dataset.word2idx[w], rel, top_k=5):
            print(f"  - {dataset.idx2word[i]:<15} ({p*100:.1f}%)")
    elif args.predict:
        tokens = args.predict.lower().split()
        if not all(t in dataset.word2idx for t in tokens):
            print("Words not in vocabulary.")
            sys.exit(1)
        indices = [dataset.word2idx[t] for t in tokens]
        for i, p in model.predict_next(indices, top_k=5):
            print(f"  - {dataset.idx2word[i]:<15} (p={p:.4f})")
    elif args.similarity:
        w1, w2 = args.similarity[0].lower(), args.similarity[1].lower()
        if w1 not in dataset.word2idx or w2 not in dataset.word2idx:
            print("Words not in vocabulary.")
            sys.exit(1)
        cos = cosine_similarity(model.E[dataset.word2idx[w1]], model.E[dataset.word2idx[w2]])
        print(f"cos({w1}, {w2}) = {cos:+.4f}")
    else:
        run_demo(model, dataset)


if __name__ == "__main__":
    main()
