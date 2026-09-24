"""
Raw NumPy Implementation of Word Embedding and Word-to-Word Transformations.
Zero external dependencies other than NumPy.

Adheres strictly to the paper:
'Word Embedding and Word-to-Word Transformations'

Implements:
- Section 2: One-hot representation x_i in R^V
- Section 3: Word Encoder matrix E in R^(V x d), e = x^T E
- Section 5 & 6: Semantic Transitions Tsyn in R^(d x d) (Tsyn ≈ I), Tant in R^(d x d) (Tant ≈ -I)
- Section 7: General Semantic Transitions Tr in R^(d x d)
- Section 8: Word Decoder matrix D in R^(d x V), s = z D, p = softmax(s)
- Section 9: Analytical gradients dL/dE, dL/dT, dL/dD and gradient descent
- Section 10-12: Recurrent Transition h_t = tanh([h_{t-1}, e_t] T_R + b) with Backpropagation Through Time (BPTT)
- Section 13-15: Unified Encode -> Transform -> Decode architecture
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes cosine similarity between two 1D vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


class RawUnifiedWordModel:
    """
    Unified Word Model implemented in pure NumPy.
    Learns E, T_r (including Tsyn and Tant), T_R, b, and D jointly via backpropagation.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        relation_names: Optional[List[str]] = None,
        seed: int = 42
    ):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.d = embed_dim
        self.h = embed_dim  # Recurrent hidden dimension h = d for unified decoder D

        if relation_names is None:
            relation_names = ["synonym", "antonym"]
        self.relation_names = relation_names

        # 1. Encoder E in R^(V x d) (Section 3)
        scale_e = np.sqrt(2.0 / (vocab_size + self.d))
        self.E = np.random.randn(vocab_size, self.d) * scale_e

        # 2. Semantic Transitions T_r in R^(d x d) (Section 5, 6, 7)
        self.T: Dict[str, np.ndarray] = {}
        for r in relation_names:
            if r == "synonym":
                # Tsyn initialized close to Identity I (Section 5)
                self.T[r] = np.eye(self.d) + np.random.randn(self.d, self.d) * 0.02
            elif r == "antonym":
                # Tant initialized close to -Identity -I (Section 6)
                self.T[r] = -np.eye(self.d) + np.random.randn(self.d, self.d) * 0.02
            else:
                scale_r = np.sqrt(2.0 / (2 * self.d))
                self.T[r] = np.random.randn(self.d, self.d) * scale_r

        # 3. Recurrent Transition T_R in R^((h+d) x h) and bias b in R^(1 x h) (Section 10)
        scale_rnn = np.sqrt(2.0 / (self.h + self.d + self.h))
        self.T_R = np.random.randn(self.h + self.d, self.h) * scale_rnn
        self.b = np.zeros((1, self.h))

        # 4. Decoder D in R^(d x V) (Section 8)
        scale_d = np.sqrt(2.0 / (self.d + vocab_size))
        self.D = np.random.randn(self.d, vocab_size) * scale_d

        # Adam optimizer moments
        self.step_count = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def adam_step(self, grads: Dict[str, np.ndarray], lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """Adam optimizer step implemented in pure NumPy."""
        self.step_count += 1
        for name, grad in grads.items():
            if name not in self.m:
                self.m[name] = np.zeros_like(grad)
                self.v[name] = np.zeros_like(grad)
            
            # Clip gradient norm for numerical stability
            gnorm = np.linalg.norm(grad)
            if gnorm > 5.0:
                grad = grad * (5.0 / gnorm)

            self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * (grad ** 2)

            m_hat = self.m[name] / (1.0 - beta1 ** self.step_count)
            v_hat = self.v[name] / (1.0 - beta2 ** self.step_count)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)

            if name == "E":
                self.E -= update
            elif name == "D":
                self.D -= update
            elif name == "T_R":
                self.T_R -= update
            elif name == "b":
                self.b -= update
            elif name.startswith("T_"):
                rel = name[2:]
                self.T[rel] -= update

    # ------------------------------------------------------------------
    # Forward Pass 1: Word-to-Word Semantic Transformation (Section 4, 15.1)
    # ------------------------------------------------------------------
    def forward_semantic(self, input_idx: int, relation: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        x -> e = x^T E -> z = e T_r -> s = z D -> p = softmax(s)
        Returns:
            e: embedding in R^(1 x d)
            z: transformed embedding in R^(1 x d)
            s: vocabulary scores in R^(1 x V)
            p: probabilities in R^(1 x V)
        """
        e = self.E[input_idx:input_idx + 1]  # shape (1, d)
        Tr = self.T[relation]
        z = np.dot(e, Tr)                   # shape (1, d)
        s = np.dot(z, self.D)                # shape (1, V)
        p = softmax(s)                      # shape (1, V)
        return e, z, s, p

    # ------------------------------------------------------------------
    # Forward Pass 2: Multiword Sequence Prediction (Section 10-12, 15.3)
    # ------------------------------------------------------------------
    def forward_sequence(
        self,
        sequence_indices: List[int]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        """
        w1...wt -> E -> RNN -> D -> softmax -> p_{t+1}
        Returns:
            states: list of recurrent memory states [h1, ..., ht]
            combined: list of concatenated inputs [u1, ..., ut] where uk = [h_{k-1}, ek]
            s: vocabulary scores in R^(1 x V)
            p: next-word probabilities in R^(1 x V)
        """
        h_t = np.zeros((1, self.h))
        states = []
        combined_inputs = []

        for w_idx in sequence_indices:
            e_t = self.E[w_idx:w_idx + 1]                 # shape (1, d)
            u_t = np.concatenate([h_t, e_t], axis=-1)      # shape (1, h + d)
            combined_inputs.append(u_t)
            h_t = np.tanh(np.dot(u_t, self.T_R) + self.b)   # shape (1, h)
            states.append(h_t)

        s = np.dot(h_t, self.D)
        p = softmax(s)
        return states, combined_inputs, s, p

    # ------------------------------------------------------------------
    # Vectorized Batched Training Steps
    # ------------------------------------------------------------------
    def train_batch_semantic(
        self,
        inputs: List[int],
        targets: List[int],
        relation: str,
        lr: float = 0.01,
        lambda_syn: float = 0.5,
        lambda_ant: float = 0.5
    ) -> float:
        """
        Vectorized mini-batch training for word-to-word semantic transition.
        Computes exact analytical gradients for E, T_r, and D.
        """
        B = len(inputs)
        inp_arr = np.array(inputs, dtype=np.int32)
        tgt_arr = np.array(targets, dtype=np.int32)

        # Forward
        E_batch = self.E[inp_arr]               # (B, d)
        Tr = self.T[relation]                   # (d, d)
        Z_batch = np.dot(E_batch, Tr)           # (B, d)
        S_batch = np.dot(Z_batch, self.D)       # (B, V)
        P_batch = softmax(S_batch)              # (B, V)

        # Cross-entropy loss: L = - 1/B sum log P[b, target[b]]
        correct_probs = np.clip(P_batch[np.arange(B), tgt_arr], 1e-12, 1.0)
        ce_loss = -float(np.mean(np.log(correct_probs)))

        # Output score error delta_S = (P - Y) / B
        delta_S = P_batch.copy()
        delta_S[np.arange(B), tgt_arr] -= 1.0
        delta_S /= B

        # Gradients
        grad_D = np.dot(Z_batch.T, delta_S)      # (d, V)
        delta_Z = np.dot(delta_S, self.D.T)      # (B, d)
        grad_Tr = np.dot(E_batch.T, delta_Z)     # (d, d)
        delta_E = np.dot(delta_Z, Tr.T)          # (B, d)

        grad_E = np.zeros_like(self.E)
        np.add.at(grad_E, inp_arr, delta_E)

        # Semantic geometric constraints (Section 5.1 & 6.1)
        E_target = self.E[tgt_arr]
        I = np.eye(self.d)

        if relation == "synonym":
            # L_syn = sum ||ei - ej||^2 / B -> grad_ei = 2*(ei - ej)/B
            diff = (E_batch - E_target) / B
            np.add.at(grad_E, inp_arr, lambda_syn * diff)
            np.add.at(grad_E, tgt_arr, -lambda_syn * diff)
            # Regularize Tsyn towards I: ||Tsyn - I||_F^2
            grad_Tr += lambda_syn * 0.1 * (Tr - I)

        elif relation == "antonym":
            # L_ant = sum ||ei + ej||^2 / B -> grad_ei = 2*(ei + ej)/B
            summ = (E_batch + E_target) / B
            np.add.at(grad_E, inp_arr, lambda_ant * summ)
            np.add.at(grad_E, tgt_arr, lambda_ant * summ)
            # Regularize Tant towards -I: ||Tant - (-I)||_F^2
            grad_Tr += lambda_ant * 0.1 * (Tr + I)

        # Optimizer step
        grads = {
            "E": grad_E,
            "D": grad_D,
            f"T_{relation}": grad_Tr
        }
        self.adam_step(grads, lr=lr)
        return ce_loss

    def train_batch_sequence(
        self,
        prefixes: List[List[int]],
        targets: List[int],
        lr: float = 0.01
    ) -> float:
        """
        Vectorized Backpropagation Through Time (BPTT) for sequences of equal length.
        """
        B = len(prefixes)
        T_len = len(prefixes[0])
        pref_arr = np.array(prefixes, dtype=np.int32) # (B, T_len)
        tgt_arr = np.array(targets, dtype=np.int32)   # (B,)

        # Forward unroll
        H_t = np.zeros((B, self.h))
        states = []
        combined_inputs = []

        for k in range(T_len):
            E_k = self.E[pref_arr[:, k]]                     # (B, d)
            U_k = np.concatenate([H_t, E_k], axis=-1)         # (B, h + d)
            combined_inputs.append(U_k)
            H_t = np.tanh(np.dot(U_k, self.T_R) + self.b)     # (B, h)
            states.append(H_t)

        S_batch = np.dot(H_t, self.D)
        P_batch = softmax(S_batch)

        correct_probs = np.clip(P_batch[np.arange(B), tgt_arr], 1e-12, 1.0)
        loss = -float(np.mean(np.log(correct_probs)))

        delta_S = P_batch.copy()
        delta_S[np.arange(B), tgt_arr] -= 1.0
        delta_S /= B

        # Gradient w.r.t Decoder D
        grad_D = np.dot(H_t.T, delta_S)

        # Gradient w.r.t final hidden state
        delta_H = np.dot(delta_S, self.D.T)

        grad_T_R = np.zeros_like(self.T_R)
        grad_b = np.zeros_like(self.b)
        grad_E = np.zeros_like(self.E)

        # BPTT
        for k in reversed(range(T_len)):
            H_k = states[k]
            U_k = combined_inputs[k]
            delta_A = delta_H * (1.0 - H_k ** 2)

            grad_b += np.sum(delta_A, axis=0, keepdims=True)
            grad_T_R += np.dot(U_k.T, delta_A)

            delta_U = np.dot(delta_A, self.T_R.T)
            delta_H = delta_U[:, :self.h]
            delta_E = delta_U[:, self.h:]

            np.add.at(grad_E, pref_arr[:, k], delta_E)

        grads = {
            "E": grad_E,
            "D": grad_D,
            "T_R": grad_T_R,
            "b": grad_b
        }
        self.adam_step(grads, lr=lr)
        return loss

    # ------------------------------------------------------------------
    # High-Level Predictions
    # ------------------------------------------------------------------
    def predict_semantic(self, input_idx: int, relation: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Word -> E -> T_r -> D -> Softmax -> Top-K candidates."""
        _, _, _, p = self.forward_semantic(input_idx, relation)
        probs = p[0]
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [(int(idx), float(probs[idx])) for idx in top_indices]

    def predict_next(self, sequence_indices: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        """Sequence -> E -> RNN -> D -> Softmax -> Top-K next-word candidates."""
        _, _, _, p = self.forward_sequence(sequence_indices)
        probs = p[0]
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [(int(idx), float(probs[idx])) for idx in top_indices]
