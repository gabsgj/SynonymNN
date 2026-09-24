"""
Training pipeline for Unified Word Embedding and Word-to-Word Transformations.
Learns:
1. Encoder E
2. Semantic transitions T_r (including Tsyn and Tant) with geometric regularizations
3. Recurrent transition T_R and bias b
4. Decoder D
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.optim as optim

from dataset import SemanticDataset
from model import UnifiedWordModel


def train_epoch(
    model: UnifiedWordModel,
    semantic_data: List[Tuple[int, int, str]],
    sequence_data: List[Tuple[List[int], int]],
    optimizer: optim.Optimizer,
    device: torch.device,
    batch_size: int = 32,
    lambda_syn: float = 0.5,
    lambda_ant: float = 0.5,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_sem_loss = 0.0
    total_seq_loss = 0.0
    sem_steps = 0
    seq_steps = 0

    # Shuffle datasets
    random.shuffle(semantic_data)
    random.shuffle(sequence_data)

    # 1. Train on Semantic Word-to-Word Transformations
    # Group by relation for batched tensor operations
    relation_batches: Dict[str, List[Tuple[int, int]]] = {}
    for inp, tgt, rel in semantic_data:
        relation_batches.setdefault(rel, []).append((inp, tgt))

    for rel, pairs in relation_batches.items():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            inputs = torch.tensor([p[0] for p in batch], dtype=torch.long, device=device)
            targets = torch.tensor([p[1] for p in batch], dtype=torch.long, device=device)

            optimizer.zero_grad()
            loss, metrics = model.compute_semantic_loss(
                inputs, targets, rel, lambda_syn=lambda_syn, lambda_ant=lambda_ant
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_sem_loss += loss.item()
            sem_steps += 1

    # 2. Train on Multiword-to-Next-Word Sequences
    # Group sequences by length for batching
    seq_by_len: Dict[int, List[Tuple[List[int], int]]] = {}
    for prefix, target in sequence_data:
        seq_by_len.setdefault(len(prefix), []).append((prefix, target))

    for length, seq_pairs in seq_by_len.items():
        for i in range(0, len(seq_pairs), batch_size):
            batch = seq_pairs[i:i + batch_size]
            prefixes = torch.tensor([p[0] for p in batch], dtype=torch.long, device=device)
            targets = torch.tensor([p[1] for p in batch], dtype=torch.long, device=device)

            optimizer.zero_grad()
            loss, loss_val = model.compute_sequence_loss(prefixes, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_seq_loss += loss.item()
            seq_steps += 1

    return {
        "avg_loss": total_loss / max(1, sem_steps + seq_steps),
        "semantic_loss": total_sem_loss / max(1, sem_steps),
        "sequence_loss": total_seq_loss / max(1, seq_steps),
    }


def evaluate(
    model: UnifiedWordModel,
    dataset: SemanticDataset,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates accuracy and geometric metrics across all tasks."""
    model.eval()

    # Pre-build lookup of valid targets for each (word, relation)
    # This accounts for words having multiple valid synonyms (e.g. happy -> joyful, glad, pleased)
    valid_semantic: Dict[Tuple[str, str], Set[str]] = {}
    for w1, w2 in dataset.synonyms:
        valid_semantic.setdefault((w1, "synonym"), set()).add(w2)
    for w1, w2 in dataset.antonyms:
        valid_semantic.setdefault((w1, "antonym"), set()).add(w2)
    for rel_name, pairs in dataset.relations.items():
        for w1, w2 in pairs:
            valid_semantic.setdefault((w1, rel_name), set()).add(w2)

    # Pre-build valid next-word completions for sequence prefixes
    valid_seq_targets: Dict[Tuple[int, ...], Set[int]] = {}
    for prefix, target in dataset.get_sequence_examples():
        valid_seq_targets.setdefault(tuple(prefix), set()).add(target)

    # 1. Evaluate Synonyms
    syn_correct = 0
    syn_inputs = {w1 for w1, _ in dataset.synonyms}
    with torch.no_grad():
        for w1 in syn_inputs:
            idx1 = dataset.vocab.get_idx(w1)
            preds = model.predict_semantic(idx1, "synonym", top_k=1)
            pred_word = dataset.vocab.get_word(preds[0][0])
            if pred_word in valid_semantic.get((w1, "synonym"), set()):
                syn_correct += 1

    # 2. Evaluate Antonyms
    ant_correct = 0
    ant_inputs = {w1 for w1, _ in dataset.antonyms}
    with torch.no_grad():
        for w1 in ant_inputs:
            idx1 = dataset.vocab.get_idx(w1)
            preds = model.predict_semantic(idx1, "antonym", top_k=1)
            pred_word = dataset.vocab.get_word(preds[0][0])
            if pred_word in valid_semantic.get((w1, "antonym"), set()):
                ant_correct += 1

    # 3. Evaluate General Relations
    rel_correct = 0
    rel_total = 0
    with torch.no_grad():
        for (w1, rel), valid_set in valid_semantic.items():
            if rel in ["synonym", "antonym"]:
                continue
            idx1 = dataset.vocab.get_idx(w1)
            preds = model.predict_semantic(idx1, rel, top_k=1)
            pred_word = dataset.vocab.get_word(preds[0][0])
            if pred_word in valid_set:
                rel_correct += 1
            rel_total += 1

    # 4. Evaluate Sequence Completions
    seq_correct = 0
    with torch.no_grad():
        for prefix_tuple, valid_targets in valid_seq_targets.items():
            preds = model.predict_next(list(prefix_tuple), top_k=1)
            if preds[0][0] in valid_targets:
                seq_correct += 1

    # 5. Geometric cosine similarities
    with torch.no_grad():
        syn_cos_list = []
        for w1, w2 in dataset.synonyms[:50]:
            e1 = model.encoder.get_embedding(dataset.vocab.get_idx(w1))
            e2 = model.encoder.get_embedding(dataset.vocab.get_idx(w2))
            cos = torch.cosine_similarity(e1, e2).item()
            syn_cos_list.append(cos)

        ant_cos_list = []
        for w1, w2 in dataset.antonyms[:50]:
            e1 = model.encoder.get_embedding(dataset.vocab.get_idx(w1))
            e2 = model.encoder.get_embedding(dataset.vocab.get_idx(w2))
            cos = torch.cosine_similarity(e1, e2).item()
            ant_cos_list.append(cos)

    return {
        "syn_acc": syn_correct / max(1, len(syn_inputs)),
        "ant_acc": ant_correct / max(1, len(ant_inputs)),
        "rel_acc": rel_correct / max(1, rel_total),
        "seq_acc": seq_correct / max(1, len(valid_seq_targets)),
        "avg_syn_cos": sum(syn_cos_list) / max(1, len(syn_cos_list)),
        "avg_ant_cos": sum(ant_cos_list) / max(1, len(ant_cos_list)),
    }


def train_model(
    epochs: int = 150,
    embed_dim: int = 32,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    save_path: str = "model_weights.pt",
    verbose: bool = True
) -> Tuple[UnifiedWordModel, SemanticDataset]:
    # Select device (prefer MPS on Apple Silicon if available, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"Using device: {device}")

    dataset = SemanticDataset("dataset.json")
    if verbose:
        print(dataset.summary())

    model = UnifiedWordModel(
        vocab_size=len(dataset.vocab),
        embed_dim=embed_dim,
        hidden_dim=embed_dim,
        relation_names=dataset.relation_names
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    semantic_pairs = dataset.get_semantic_pairs()
    sequence_pairs = dataset.get_sequence_examples()

    if verbose:
        print("\nStarting Unified Training...")
        print(f"{'Epoch':<8}{'Total Loss':<14}{'Sem Loss':<12}{'Seq Loss':<12}{'Syn Acc':<10}{'Ant Acc':<10}{'Seq Acc':<10}{'Syn Cos':<10}{'Ant Cos':<10}")
        print("-" * 90)

    for epoch in range(1, epochs + 1):
        losses = train_epoch(
            model=model,
            semantic_data=semantic_pairs,
            sequence_data=sequence_pairs,
            optimizer=optimizer,
            device=device,
            batch_size=32,
            lambda_syn=0.5,
            lambda_ant=0.5
        )
        scheduler.step()

        if epoch % 25 == 0 or epoch == 1 or epoch == epochs:
            metrics = evaluate(model, dataset, device)
            if verbose:
                print(
                    f"{epoch:<8}"
                    f"{losses['avg_loss']:<14.4f}"
                    f"{losses['semantic_loss']:<12.4f}"
                    f"{losses['sequence_loss']:<12.4f}"
                    f"{metrics['syn_acc']*100:<10.1f}%"
                    f"{metrics['ant_acc']*100:<10.1f}%"
                    f"{metrics['seq_acc']*100:<10.1f}%"
                    f"{metrics['avg_syn_cos']:<10.2f}"
                    f"{metrics['avg_ant_cos']:<10.2f}"
                )

    # Save weights and vocabulary metadata
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_word2idx": dataset.vocab.word2idx,
        "relation_names": dataset.relation_names,
        "embed_dim": embed_dim,
    }
    torch.save(checkpoint, save_path)
    if verbose:
        print(f"\nModel checkpoint saved to {save_path}")

    return model, dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unified Word Embedding and Transformation Model")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--embed_dim", type=int, default=32, help="Embedding dimension d")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--save", type=str, default="model_weights.pt", help="Checkpoint save path")
    args = parser.parse_args()

    train_model(epochs=args.epochs, embed_dim=args.embed_dim, lr=args.lr, save_path=args.save)
