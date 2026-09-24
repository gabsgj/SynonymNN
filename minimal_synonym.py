#!/usr/bin/env python3
"""
Minimal Word-to-Synonym Neural Program from Scratch
Based directly on Sections 2, 3, 4, 5, 8, 9 of the paper:
'Word Embedding and Word-to-Word Transformations'

Architecture:
  Word (index i)
       |
       v  [One-Hot x_i in R^V]
  1. Encoder E in R^(V x d)
       |
       v  [Embedding e = x^T E in R^(1 x d)]
  2. Transition Tsyn in R^(d x d)
       |
       v  [Transformed z = e Tsyn in R^(1 x d)]
  3. Decoder D in R^(d x V)
       |
       v  [Scores s = z D in R^(1 x V)]
  4. Softmax
       |
       v  [Probabilities p in R^(1 x V)]
  Predicted Synonym Word
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Dataset & Vocabulary (Small & Transparent)
# ---------------------------------------------------------------------------
words = ["happy", "joyful", "glad", "sad", "hot", "cold"]
V = len(words)
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for i, w in enumerate(words)}

# Synonym training pairs: (input_word, target_synonym)
synonym_pairs = [
    ("happy", "joyful"),
    ("joyful", "happy"),
    ("happy", "glad"),
    ("glad", "happy"),
    ("joyful", "glad"),
    ("glad", "joyful"),
]

# ---------------------------------------------------------------------------
# 2. Model Parameters (2D Embeddings so we can print & see them directly!)
# ---------------------------------------------------------------------------
d = 2  # 2D continuous embedding space
np.random.seed(42)

# E: Encoder matrix (V x d) - each row i is word i's embedding e_i
E = np.random.randn(V, d) * 0.5

# Tsyn: Transition matrix (d x d) - initialized near Identity (Section 5)
Tsyn = np.eye(d) + np.random.randn(d, d) * 0.05

# D: Decoder matrix (d x V) - maps 2D vector back to V vocabulary scores
D = np.random.randn(d, V) * 0.5


# ---------------------------------------------------------------------------
# 3. Forward Pass Functions (Sections 3, 4, 8)
# ---------------------------------------------------------------------------
def softmax(scores):
    """Computes p_j = exp(s_j) / sum_k exp(s_k) with numerical stability."""
    exp_s = np.exp(scores - np.max(scores))
    return exp_s / np.sum(exp_s)


def forward(word_idx):
    """
    Complete forward computation (Section 4):
      x -> e = x^T E -> z = e Tsyn -> s = z D -> p = softmax(s)
    """
    # Step 1: e = x^T E (selecting row i of E)
    e = E[word_idx:word_idx + 1]       # shape: (1, 2)

    # Step 2: z = e Tsyn (transforming representation)
    z = np.dot(e, Tsyn)                # shape: (1, 2)

    # Step 3: s = z D (scoring against all vocabulary words)
    s = np.dot(z, D)[0]                # shape: (V,)

    # Step 4: p = softmax(s) (probabilities over vocabulary)
    p = softmax(s)                     # shape: (V,)

    return e, z, s, p


# ---------------------------------------------------------------------------
# 4. Training Loop: Analytical Gradient Descent (Section 9)
# ---------------------------------------------------------------------------
def train(epochs=150, lr=0.08, lambda_syn=0.3):
    global E, Tsyn, D

    print("=" * 65)
    print(" TRAINING MINIMAL SYNONYM NETWORK")
    print(f" Vocabulary: {words} (V={V}, d={d})")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for w_in, w_tgt in synonym_pairs:
            i = word2idx[w_in]
            c = word2idx[w_tgt]

            # Forward pass
            e, z, s, p = forward(i)

            # Cross-entropy loss: L_CE = -log p_c
            loss_ce = -np.log(max(p[c], 1e-12))

            # Semantic loss: L_syn = 0.5 * ||e_i - e_c||^2 (Section 5.1)
            e_tgt = E[c:c + 1]
            loss_syn = 0.5 * np.sum((e - e_tgt) ** 2)

            loss = loss_ce + lambda_syn * loss_syn
            total_loss += loss

            # --- BACKPROPAGATION (Section 9) ---
            # 1. dL/ds = p - y (where y is one-hot target vector)
            delta_s = p.copy()
            delta_s[c] -= 1.0          # shape: (V,)
            delta_s = delta_s.reshape(1, V)

            # 2. dL/dD = z^T (dL/ds)
            grad_D = np.dot(z.T, delta_s)

            # 3. dL/dz = (dL/ds) D^T
            delta_z = np.dot(delta_s, D.T)

            # 4. dL/dTsyn = e^T (dL/dz) + regularizer towards Identity
            grad_Tsyn = np.dot(e.T, delta_z) + 0.05 * (Tsyn - np.eye(d))

            # 5. dL/de = (dL/dz) Tsyn^T
            delta_e = np.dot(delta_z, Tsyn.T)

            # 6. Accumulate into encoder rows with semantic distance gradient
            diff = (e - e_tgt)
            grad_E_i = delta_e + lambda_syn * diff
            grad_E_c = -lambda_syn * diff

            # --- GRADIENT DESCENT UPDATES ---
            D -= lr * grad_D
            Tsyn -= lr * grad_Tsyn
            E[i:i + 1] -= lr * grad_E_i
            E[c:c + 1] -= lr * grad_E_c

        if epoch % 30 == 0 or epoch == 1:
            avg_loss = total_loss / len(synonym_pairs)
            print(f"Epoch {epoch:<3} | Avg Loss: {avg_loss:.4f}")

    print("\nTraining complete!\n")


# ---------------------------------------------------------------------------
# 5. Demonstration & Vector Inspection
# ---------------------------------------------------------------------------
def demonstrate():
    print("=" * 65)
    print(" 1. LEARNED 2D WORD EMBEDDINGS (E matrix)")
    print("=" * 65)
    for w in words:
        idx = word2idx[w]
        vec = E[idx]
        print(f"  {w:<8} -> e = [{vec[0]:+7.3f}, {vec[1]:+7.3f}]")

    print("\n" + "=" * 65)
    print(" 2. COSINE SIMILARITIES (Section 5: ei ≈ ej for synonyms)")
    print("=" * 65)
    def cos_sim(w1, w2):
        v1, v2 = E[word2idx[w1]], E[word2idx[w2]]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    print(f"  cos(happy, joyful) = {cos_sim('happy', 'joyful'):+.4f}  (Synonym: clustered together!)")
    print(f"  cos(happy, glad)   = {cos_sim('happy', 'glad'):+.4f}  (Synonym: clustered together!)")
    print(f"  cos(happy, sad)    = {cos_sim('happy', 'sad'):+.4f}  (Non-synonym)")
    print(f"  cos(happy, cold)   = {cos_sim('happy', 'cold'):+.4f}  (Unrelated)")

    print("\n" + "=" * 65)
    print(" 3. LEARNED SYNONYM TRANSITION MATRIX Tsyn (Section 5: Tsyn ≈ I)")
    print("=" * 65)
    print(f"  Tsyn =\n{np.array2string(Tsyn, prefix='    ', precision=4)}")
    print(f"  Frobenius distance ||Tsyn - I||: {np.linalg.norm(Tsyn - np.eye(d)):.4f}")

    print("\n" + "=" * 65)
    print(" 4. SYNONYM PREDICTION DEMO (w -> E -> Tsyn -> D -> softmax)")
    print("=" * 65)
    for test_word in ["happy", "joyful", "glad"]:
        idx = word2idx[test_word]
        _, _, _, p = forward(idx)

        # Sort predictions by probability
        ranked_indices = np.argsort(p)[::-1]
        top_preds = [f"{idx2word[i]} ({p[i]*100:.1f}%)" for i in ranked_indices[:3]]
        print(f"  Input: '{test_word:<6}' -> Top Predictions: {', '.join(top_preds)}")

    print("\n" + "=" * 65)
    print(" 5. 2D ASCII VISUALIZATION OF WORD EMBEDDINGS")
    print("=" * 65)
    # Simple 21x41 text grid to visualize vector clustering
    grid_h, grid_w = 15, 35
    grid = [[" " for _ in range(grid_w)] for _ in range(grid_h)]

    # Normalize vectors to fit in grid
    min_x, max_x = E[:, 0].min(), E[:, 0].max()
    min_y, max_y = E[:, 1].min(), E[:, 1].max()

    for w in words:
        idx = word2idx[w]
        x, y = E[idx]
        col = int((x - min_x) / (max_x - min_x + 1e-8) * (grid_w - 8))
        row = int((y - min_y) / (max_y - min_y + 1e-8) * (grid_h - 1))
        # Place word label on grid
        for ch_idx, ch in enumerate(w):
            if col + ch_idx < grid_w:
                grid[row][col + ch_idx] = ch

    print("  +-" + "-" * grid_w + "-+")
    for r in reversed(range(grid_h)):
        print(f"  | {''.join(grid[r])} |")
    print("  +-" + "-" * grid_w + "-+")
    print("  Notice how ('happy', 'joyful', 'glad') cluster in the same area!\n")


if __name__ == "__main__":
    train(epochs=150, lr=0.08, lambda_syn=0.3)
    demonstrate()
