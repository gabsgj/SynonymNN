#!/usr/bin/env python3
"""
Main CLI Application for Word Embedding and Word-to-Word Transformations.
Adheres strictly to the principles and formulas in:
'Word Embedding and Word-to-Word Transformations'

Usage:
    python main.py --demo
    python main.py --interactive
    python main.py --synonym <word>
    python main.py --antonym <word>
    python main.py --relation <rel_name> <word>
    python main.py --predict "<prefix sentence>"
    python main.py --similarity <word1> <word2>
    python main.py --train [--epochs N]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

from dataset import SemanticDataset
from model import UnifiedWordModel
from train import train_model


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_or_train_model(
    model_path: str = "model_weights.pt",
    dataset_path: str = "dataset.json",
    device: Optional[torch.device] = None,
    epochs_if_missing: int = 60,
    embed_dim: int = 32
) -> Tuple[UnifiedWordModel, SemanticDataset]:
    """Loads existing model checkpoint, or trains from scratch if not present."""
    if device is None:
        device = get_device()

    dataset = SemanticDataset(dataset_path)

    if not os.path.exists(model_path):
        print(f"[*] Checkpoint '{model_path}' not found. Training a new model...")
        model, _ = train_model(
            epochs=epochs_if_missing,
            embed_dim=embed_dim,
            save_path=model_path,
            verbose=True
        )
        return model, dataset

    checkpoint = torch.load(model_path, map_location=device)
    dim = checkpoint.get("embed_dim", embed_dim)
    rel_names = checkpoint.get("relation_names", dataset.relation_names)

    model = UnifiedWordModel(
        vocab_size=len(dataset.vocab),
        embed_dim=dim,
        hidden_dim=dim,
        relation_names=rel_names
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, dataset


def display_banner():
    banner = """
=============================================================================
     WORD EMBEDDING & WORD-TO-WORD TRANSFORMATIONS (NEURAL SYSTEM)
  Discrete Word -> Encoding (E) -> Transition (T) -> Decoding (D) -> Softmax
=============================================================================
"""
    print(banner)


def run_demo(model: UnifiedWordModel, dataset: SemanticDataset):
    """Runs a complete walkthrough of all sections and equations in the paper."""
    display_banner()
    vocab = dataset.vocab
    device = next(model.parameters()).device

    print("\n" + "=" * 75)
    print("1. ARCHITECTURE OVERVIEW (Section 1, 14, 15)")
    print("=" * 75)
    print(f"Vocabulary Size (V):     {len(vocab)}")
    print(f"Embedding Dimension (d):  {model.embed_dim}")
    print(f"Recurrent Hidden Dim (h): {model.hidden_dim}")
    print(f"Encoder Matrix E:        shape {tuple(model.encoder.E.shape)}")
    print(f"Decoder Matrix D:        shape {tuple(model.decoder.D.shape)}")
    print(f"Recurrent Operator T_R:  shape {tuple(model.rnn_transition.T_R.shape)}")
    print(f"Semantic Operators T_r:  {list(model.semantic_transition.transitions.keys())}")

    print("\n" + "=" * 75)
    print("2. WORD-TO-SYNONYM TRANSITION (Section 5: Tsyn ≈ I, ei ≈ ej)")
    print("=" * 75)
    test_synonyms = ["happy", "cold", "fast", "bright", "brave", "clean", "smart"]
    for word in test_synonyms:
        if word not in vocab:
            continue
        idx = vocab.get_idx(word)
        preds = model.predict_semantic(idx, "synonym", top_k=4)
        pred_strs = [f"{vocab.get_word(p_idx)} ({prob*100:.1f}%)" for p_idx, prob in preds]
        print(f"  Input: '{word:<8}' -> Synonyms: {', '.join(pred_strs)}")

    # Matrix inspection
    T_syn = model.semantic_transition.get_matrix("synonym")
    I = torch.eye(model.embed_dim, device=device)
    diff_syn = torch.norm(T_syn - I, p="fro").item()
    diag_syn = torch.diag(T_syn).mean().item()
    print(f"\n  [Mathematical Check] Tsyn Frobenius distance to Identity ||Tsyn - I||_F: {diff_syn:.4f}")
    print(f"  [Mathematical Check] Tsyn Average Diagonal Entry: {diag_syn:.4f} (theoretical: 1.0)")

    print("\n" + "=" * 75)
    print("3. WORD-TO-ANTONYM TRANSITION (Section 6: Tant ≈ -I, ei ≈ -ej)")
    print("=" * 75)
    test_antonyms = ["hot", "happy", "fast", "big", "dark", "strong", "peace", "clean"]
    for word in test_antonyms:
        if word not in vocab:
            continue
        idx = vocab.get_idx(word)
        preds = model.predict_semantic(idx, "antonym", top_k=4)
        pred_strs = [f"{vocab.get_word(p_idx)} ({prob*100:.1f}%)" for p_idx, prob in preds]
        print(f"  Input: '{word:<8}' -> Antonyms: {', '.join(pred_strs)}")

    # Matrix inspection
    T_ant = model.semantic_transition.get_matrix("antonym")
    neg_I = -torch.eye(model.embed_dim, device=device)
    diff_ant = torch.norm(T_ant - neg_I, p="fro").item()
    diag_ant = torch.diag(T_ant).mean().item()
    print(f"\n  [Mathematical Check] Tant Frobenius distance to -Identity ||Tant - (-I)||_F: {diff_ant:.4f}")
    print(f"  [Mathematical Check] Tant Average Diagonal Entry: {diag_ant:.4f} (theoretical: -1.0)")

    print("\n" + "=" * 75)
    print("4. GENERAL SEMANTIC TRANSITIONS (Section 7: Tr in R^(d x d), ej ≈ ei Tr)")
    print("=" * 75)
    sample_queries = [
        ("france", "country_to_capital"),
        ("japan", "country_to_capital"),
        ("cat", "singular_to_plural"),
        ("child", "singular_to_plural"),
        ("walk", "present_to_past"),
        ("drink", "present_to_past"),
        ("animal", "category_to_instance"),
        ("fruit", "category_to_instance"),
    ]
    for word, rel in sample_queries:
        if word not in vocab or rel not in model.semantic_transition.transitions:
            continue
        idx = vocab.get_idx(word)
        preds = model.predict_semantic(idx, rel, top_k=3)
        pred_strs = [f"{vocab.get_word(p_idx)} ({prob*100:.1f}%)" for p_idx, prob in preds]
        print(f"  Relation [{rel}]: '{word}' -> {', '.join(pred_strs)}")

    print("\n" + "=" * 75)
    print("5. MULTIWORD-TO-WORD PREDICTION (Section 10, 11, 12, 15.3, 15.4)")
    print("   Example from PDF Section 12: 'The cat drinks' -> 'milk'")
    print("=" * 75)
    test_prefixes = [
        "the cat drinks",
        "the cat eats",
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
        if not all(w in vocab for w in words):
            continue
        indices = [vocab.get_idx(w) for w in words]
        preds = model.predict_next(indices, top_k=3)
        pred_strs = [f"{vocab.get_word(p_idx)} (p={prob:.3f})" for p_idx, prob in preds]
        print(f"  Prefix: '{prefix}'")
        print(f"    -> Next Word Predictions: {', '.join(pred_strs)}")

    print("\n" + "=" * 75)
    print("6. AUTOREGRESSIVE GENERATION (Repeated Recurrent Transition)")
    print("=" * 75)
    seed_prompts = ["the cat", "the dog", "the sun", "the teacher", "the winter"]
    for prompt in seed_prompts:
        curr_words = prompt.lower().split()
        curr_indices = [vocab.get_idx(w) for w in curr_words]
        for _ in range(3):
            preds = model.predict_next(curr_indices, top_k=1)
            next_idx = preds[0][0]
            next_word = vocab.get_word(next_idx)
            curr_indices.append(next_idx)
            curr_words.append(next_word)
        print(f"  Generated sequence: \"{' '.join(curr_words)}\"")

    print("\n" + "=" * 75)
    print("7. COSINE SIMILARITY & GEOMETRIC VERIFICATION")
    print("=" * 75)
    sim_pairs = [
        ("happy", "joyful", "Synonym (should be near +1.0)"),
        ("hot", "cold", "Antonym (should be near -1.0)"),
        ("happy", "sad", "Antonym (should be near -1.0)"),
        ("bright", "dark", "Antonym (should be near -1.0)"),
        ("fast", "quick", "Synonym (should be near +1.0)"),
        ("cat", "milk", "Unrelated semantic words"),
    ]
    for w1, w2, label in sim_pairs:
        if w1 in vocab and w2 in vocab:
            e1 = model.encoder.get_embedding(vocab.get_idx(w1))
            e2 = model.encoder.get_embedding(vocab.get_idx(w2))
            cos = F.cosine_similarity(e1, e2, dim=-1).item()
            print(f"  cos({w1:<7}, {w2:<7}) = {cos:+.4f} | {label}")
    print("=" * 75 + "\n")


def run_interactive(model: UnifiedWordModel, dataset: SemanticDataset):
    """Interactive loop for querying the model directly."""
    display_banner()
    vocab = dataset.vocab
    relations = dataset.relation_names

    while True:
        print("\nSelect an operation:")
        print("  1. Query Synonyms of a word (Tsyn ≈ I)")
        print("  2. Query Antonyms of a word (Tant ≈ -I)")
        print("  3. Query General Semantic Relation (Tr)")
        print("  4. Predict Next Word from Sequence (RNN Transition)")
        print("  5. Word Vector Similarity Inspector")
        print("  6. Autoregressive Sentence Completion")
        print("  0. Exit")

        choice = input("\nEnter choice [0-6]: ").strip()
        if choice in ["0", "exit", "quit", "q"]:
            print("Exiting.")
            break

        if choice == "1":
            word = input("Enter word for synonyms: ").strip().lower()
            if word not in vocab:
                print(f"Error: '{word}' is not in vocabulary.")
                continue
            idx = vocab.get_idx(word)
            preds = model.predict_semantic(idx, "synonym", top_k=5)
            print(f"\nTop Synonyms for '{word}':")
            for rank, (p_idx, prob) in enumerate(preds, 1):
                print(f"  {rank}. {vocab.get_word(p_idx):<15} (confidence: {prob*100:.2f}%)")

        elif choice == "2":
            word = input("Enter word for antonyms: ").strip().lower()
            if word not in vocab:
                print(f"Error: '{word}' is not in vocabulary.")
                continue
            idx = vocab.get_idx(word)
            preds = model.predict_semantic(idx, "antonym", top_k=5)
            print(f"\nTop Antonyms for '{word}':")
            for rank, (p_idx, prob) in enumerate(preds, 1):
                print(f"  {rank}. {vocab.get_word(p_idx):<15} (confidence: {prob*100:.2f}%)")

        elif choice == "3":
            print(f"Available relations: {relations}")
            rel = input("Enter relation name: ").strip()
            if rel not in model.semantic_transition.transitions:
                print(f"Error: unknown relation '{rel}'.")
                continue
            word = input("Enter base word: ").strip().lower()
            if word not in vocab:
                print(f"Error: '{word}' is not in vocabulary.")
                continue
            idx = vocab.get_idx(word)
            preds = model.predict_semantic(idx, rel, top_k=5)
            print(f"\nTransformation [{rel}] for '{word}':")
            for rank, (p_idx, prob) in enumerate(preds, 1):
                print(f"  {rank}. {vocab.get_word(p_idx):<15} (confidence: {prob*100:.2f}%)")

        elif choice == "4":
            text = input("Enter preceding words (e.g. 'the cat drinks'): ").strip().lower()
            words = text.split()
            missing = [w for w in words if w not in vocab]
            if missing:
                print(f"Error: words not in vocabulary: {missing}")
                continue
            indices = [vocab.get_idx(w) for w in words]
            preds = model.predict_next(indices, top_k=5)
            print(f"\nNext-word predictions for '{text}':")
            for rank, (p_idx, prob) in enumerate(preds, 1):
                print(f"  {rank}. {vocab.get_word(p_idx):<15} (probability: {prob:.4f})")

        elif choice == "5":
            w1 = input("Enter first word: ").strip().lower()
            w2 = input("Enter second word: ").strip().lower()
            if w1 not in vocab or w2 not in vocab:
                print(f"Error: words must be in vocabulary.")
                continue
            e1 = model.encoder.get_embedding(vocab.get_idx(w1))
            e2 = model.encoder.get_embedding(vocab.get_idx(w2))
            cos = F.cosine_similarity(e1, e2, dim=-1).item()
            euclid = torch.norm(e1 - e2, p=2).item()
            print(f"\nGeometric Comparison between '{w1}' and '{w2}':")
            print(f"  Cosine Similarity: {cos:+.4f} (range: -1.0 to +1.0)")
            print(f"  Euclidean Distance: {euclid:.4f}")

        elif choice == "6":
            seed = input("Enter seed prompt (e.g. 'the cat'): ").strip().lower()
            words = seed.split()
            missing = [w for w in words if w not in vocab]
            if missing:
                print(f"Error: words not in vocabulary: {missing}")
                continue
            length = int(input("How many words to generate [1-10]: ").strip() or "3")
            curr_indices = [vocab.get_idx(w) for w in words]
            for _ in range(length):
                preds = model.predict_next(curr_indices, top_k=1)
                curr_indices.append(preds[0][0])
            gen_text = " ".join(vocab.get_word(i) for i in curr_indices)
            print(f"\nResult: \"{gen_text}\"")


def main():
    parser = argparse.ArgumentParser(
        description="Word Embedding and Word-to-Word Transformations"
    )
    parser.add_argument("--demo", action="store_true", help="Run full architectural demonstration")
    parser.add_argument("--interactive", action="store_true", help="Start interactive CLI session")
    parser.add_argument("--train", action="store_true", help="Train the model from scratch")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs (default: 60)")
    parser.add_argument("--embed_dim", type=int, default=32, help="Embedding dimension d (default: 32)")
    parser.add_argument("--synonym", type=str, help="Query synonyms for word")
    parser.add_argument("--antonym", type=str, help="Query antonyms for word")
    parser.add_argument("--relation", nargs=2, metavar=("RELATION", "WORD"), help="Query relation for word")
    parser.add_argument("--predict", type=str, help="Predict next word for prefix sequence")
    parser.add_argument("--similarity", nargs=2, metavar=("WORD1", "WORD2"), help="Compute cosine similarity")
    parser.add_argument("--generate", type=str, help="Generate sentence from seed prefix")

    args = parser.parse_args()

    device = get_device()

    if args.train:
        train_model(epochs=args.epochs, embed_dim=args.embed_dim, save_path="model_weights.pt")
        return

    model, dataset = load_or_train_model(
        model_path="model_weights.pt",
        dataset_path="dataset.json",
        device=device,
        epochs_if_missing=args.epochs,
        embed_dim=args.embed_dim
    )

    if args.demo:
        run_demo(model, dataset)
    elif args.interactive:
        run_interactive(model, dataset)
    elif args.synonym:
        word = args.synonym.lower()
        if word not in dataset.vocab:
            print(f"Error: '{word}' not in vocabulary.")
            sys.exit(1)
        preds = model.predict_semantic(dataset.vocab.get_idx(word), "synonym", top_k=5)
        print(f"Synonyms for '{word}':")
        for idx, prob in preds:
            print(f"  - {dataset.vocab.get_word(idx):<15} ({prob*100:.1f}%)")
    elif args.antonym:
        word = args.antonym.lower()
        if word not in dataset.vocab:
            print(f"Error: '{word}' not in vocabulary.")
            sys.exit(1)
        preds = model.predict_semantic(dataset.vocab.get_idx(word), "antonym", top_k=5)
        print(f"Antonyms for '{word}':")
        for idx, prob in preds:
            print(f"  - {dataset.vocab.get_word(idx):<15} ({prob*100:.1f}%)")
    elif args.relation:
        rel, word = args.relation[0], args.relation[1].lower()
        if word not in dataset.vocab:
            print(f"Error: '{word}' not in vocabulary.")
            sys.exit(1)
        if rel not in model.semantic_transition.transitions:
            print(f"Error: unknown relation '{rel}'. Available: {list(model.semantic_transition.transitions.keys())}")
            sys.exit(1)
        preds = model.predict_semantic(dataset.vocab.get_idx(word), rel, top_k=5)
        print(f"Relation [{rel}] for '{word}':")
        for idx, prob in preds:
            print(f"  - {dataset.vocab.get_word(idx):<15} ({prob*100:.1f}%)")
    elif args.predict:
        words = args.predict.lower().split()
        missing = [w for w in words if w not in dataset.vocab]
        if missing:
            print(f"Error: words not in vocabulary: {missing}")
            sys.exit(1)
        indices = [dataset.vocab.get_idx(w) for w in words]
        preds = model.predict_next(indices, top_k=5)
        print(f"Next-word predictions for '{args.predict}':")
        for idx, prob in preds:
            print(f"  - {dataset.vocab.get_word(idx):<15} (p={prob:.4f})")
    elif args.similarity:
        w1, w2 = args.similarity[0].lower(), args.similarity[1].lower()
        if w1 not in dataset.vocab or w2 not in dataset.vocab:
            print("Error: both words must be in vocabulary.")
            sys.exit(1)
        e1 = model.encoder.get_embedding(dataset.vocab.get_idx(w1))
        e2 = model.encoder.get_embedding(dataset.vocab.get_idx(w2))
        cos = F.cosine_similarity(e1, e2, dim=-1).item()
        print(f"Cosine similarity cos({w1}, {w2}) = {cos:+.4f}")
    elif args.generate:
        words = args.generate.lower().split()
        indices = [dataset.vocab.get_idx(w) for w in words if w in dataset.vocab]
        for _ in range(4):
            preds = model.predict_next(indices, top_k=1)
            indices.append(preds[0][0])
        print("Generated:", " ".join(dataset.vocab.get_word(i) for i in indices))
    else:
        # Default action: run demo
        run_demo(model, dataset)


if __name__ == "__main__":
    main()
