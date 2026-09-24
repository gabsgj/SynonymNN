"""
Unified Neural Network Architecture for Word Embedding and Transformations.
Implements the exact formulations from:
'Word Embedding and Word-to-Word Transformations'

Components:
1. Encoder: E in R^(V x d), computes e = x^T E
2. Semantic Transitions: T_r in R^(d x d) (T_syn approx I, T_ant approx -I, and learned T_r)
3. Recurrent Transition: T_R in R^((h+d) x h), b in R^(1 x h), h_t = tanh([h_{t-1}, e_t] T_R + b)
4. Decoder: D in R^(d x V), s = z D, p = softmax(s)
5. Unified Model: Encode -> Transform -> Decode
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class WordEncoder(nn.Module):
    """
    Encoder matrix E in R^(V x d) (Section 3).
    For word with one-hot vector x_i in R^V:
        e_i = x_i^T E  (in R^(1 x d))
    Row i of E contains the continuous embedding vector e_i.
    """
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Learned continuous embedding matrix E in R^(V x d)
        self.E = nn.Parameter(torch.empty(vocab_size, embed_dim))
        # Initialize with standard normal scaled by sqrt(2 / (V + d))
        nn.init.xavier_uniform_(self.E)

    def forward(self, x: Union[torch.Tensor, int, List[int]]) -> torch.Tensor:
        """
        Forward pass.
        If x is 2D one-hot tensor (batch_size, V): computes x @ E.
        If x is 1D tensor/list of indices: selects corresponding rows of E.
        Both are mathematically identical: e_i = x_i^T E.
        """
        if isinstance(x, (int, list)):
            x = torch.tensor(x, dtype=torch.long, device=self.E.device)

        if x.dim() == 2 and x.size(-1) == self.vocab_size:
            # Explicit one-hot matrix multiplication: x^T E
            return torch.matmul(x, self.E)
        else:
            # Row selection: selecting row i of E (Section 3)
            return self.E[x]

    def get_embedding(self, idx: int) -> torch.Tensor:
        """Returns embedding vector e_i in R^(1 x d)."""
        return self.E[idx:idx+1]


class WordDecoder(nn.Module):
    """
    Decoder matrix D in R^(d x V) (Section 8).
    For transformed representation z in R^(1 x d):
        s = z D  (vector of scores in R^(1 x V))
        p = softmax(s)
    """
    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        # Learned decoder matrix D in R^(d x V)
        self.D = nn.Parameter(torch.empty(embed_dim, vocab_size))
        nn.init.xavier_uniform_(self.D)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes vocabulary scores s = z D and probability distribution p = softmax(s).
        Args:
            z: transformed embedding tensor of shape (batch_size, d)
        Returns:
            scores: s in R^(batch_size x V)
            probs:  p in R^(batch_size x V)
        """
        s = torch.matmul(z, self.D)
        p = F.softmax(s, dim=-1)
        return s, p


class SemanticTransition(nn.Module):
    """
    Learned semantic transition operator T_r in R^(d x d) (Section 5, 6, 7).
    z = e T_r
    Supports relations:
        - synonym: Tsyn with prior Tsyn approx I
        - antonym: Tant with prior Tant approx -I
        - general relation r: Tr
    """
    def __init__(self, embed_dim: int, relation_names: List[str]):
        super().__init__()
        self.embed_dim = embed_dim
        self.relation_names = relation_names
        
        # Create a ParameterDict for transition matrices T_r in R^(d x d)
        self.transitions = nn.ParameterDict()
        for r in relation_names:
            T = nn.Parameter(torch.empty(embed_dim, embed_dim))
            if r == "synonym":
                # Initialized close to Identity matrix I (Section 5)
                nn.init.eye_(T)
                with torch.no_grad():
                    T.add_(torch.randn_like(T) * 0.02)
            elif r == "antonym":
                # Initialized close to -Identity matrix -I (Section 6)
                nn.init.eye_(T)
                with torch.no_grad():
                    T.mul_(-1.0).add_(torch.randn_like(T) * 0.02)
            else:
                # Random orthogonal / xavier initialization for general relations
                nn.init.orthogonal_(T)
            self.transitions[r] = T

    def forward(self, e: torch.Tensor, relation: str) -> torch.Tensor:
        """
        Computes z = e T_r
        Args:
            e: input embedding in R^(batch_size x d)
            relation: relation name string
        """
        if relation not in self.transitions:
            raise ValueError(f"Unknown relation '{relation}'. Available: {list(self.transitions.keys())}")
        Tr = self.transitions[relation]
        return torch.matmul(e, Tr)

    def get_matrix(self, relation: str) -> torch.Tensor:
        return self.transitions[relation]


class RecurrentTransition(nn.Module):
    """
    Recurrent sequence transition operator (Section 10, 11, 15.3).
    h_t = tanh([h_{t-1}, e_t] T_R + b)
    where:
        e_t in R^(1 x d)
        h_t in R^(1 x h)
        [h_{t-1}, e_t] in R^(1 x (h + d))
        T_R in R^((h + d) x h)
        b in R^(1 x h)
    """
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # T_R in R^((hidden_dim + embed_dim) x hidden_dim)
        self.T_R = nn.Parameter(torch.empty(hidden_dim + embed_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(1, hidden_dim))
        
        # Initialize T_R
        nn.init.xavier_uniform_(self.T_R)

    def step(self, h_prev: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        """
        Single recurrent step:
            combined = [h_{t-1}, e_t]
            h_t = tanh(combined @ T_R + b)
        """
        # Concatenate along the feature dimension: [h_{t-1}, e_t]
        combined = torch.cat([h_prev, e_t], dim=-1)
        h_t = torch.tanh(torch.matmul(combined, self.T_R) + self.b)
        return h_t

    def forward(self, embeddings: torch.Tensor, h_0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a sequence of word embeddings.
        Args:
            embeddings: tensor of shape (batch_size, seq_len, embed_dim)
            h_0: initial state of shape (batch_size, hidden_dim), defaults to zeros
        Returns:
            all_states: tensor of shape (batch_size, seq_len, hidden_dim)
            final_state: tensor of shape (batch_size, hidden_dim)
        """
        batch_size, seq_len, _ = embeddings.size()
        if h_0 is None:
            h_t = torch.zeros(batch_size, self.hidden_dim, device=embeddings.device)
        else:
            h_t = h_0

        states = []
        for t in range(seq_len):
            e_t = embeddings[:, t, :]
            h_t = self.step(h_t, e_t)
            states.append(h_t.unsqueeze(1))

        all_states = torch.cat(states, dim=1)
        return all_states, h_t


class UnifiedWordModel(nn.Module):
    """
    Unified Word Embedding and Transformation Model (Section 13, 14, 15).
    Combines:
      - Word-to-semantic-word transformation:  x -> E -> T_r -> D -> scores -> softmax -> w_hat
      - Multiword-to-next-word prediction:      w1...wt -> E -> RNN -> D -> scores -> softmax -> w_hat_{t+1}
    Both tasks share the continuous embedding space E and vocabulary decoder D.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        hidden_dim: Optional[int] = None,
        relation_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Default hidden dimension h = embed_dim for seamless unified decoder D
        self.hidden_dim = hidden_dim if hidden_dim is not None else embed_dim

        if relation_names is None:
            relation_names = ["synonym", "antonym"]

        # 1. Shared Encoder E in R^(V x d)
        self.encoder = WordEncoder(vocab_size, embed_dim)

        # 2. Semantic Transition operators T_r in R^(d x d)
        self.semantic_transition = SemanticTransition(embed_dim, relation_names)

        # 3. Recurrent Transition operator T_R in R^((h+d) x h)
        self.rnn_transition = RecurrentTransition(embed_dim, self.hidden_dim)

        # Optional hidden-to-embed projection if hidden_dim != embed_dim
        if self.hidden_dim != self.embed_dim:
            self.rnn_proj = nn.Linear(self.hidden_dim, self.embed_dim, bias=False)
        else:
            self.rnn_proj = nn.Identity()

        # 4. Shared Decoder D in R^(d x V)
        self.decoder = WordDecoder(embed_dim, vocab_size)

    # -------------------------------------------------------------
    # 1. Semantic Word-to-Word Transformation (Section 4 - 7, 15.1)
    # -------------------------------------------------------------
    def forward_semantic(
        self,
        input_indices: torch.Tensor,
        relation: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward computation (Section 4, 15.1):
            e = x^T E
            z = e T_r
            s = z D
            p = softmax(s)
        Returns:
            e: input embedding (batch_size, d)
            z: transformed embedding (batch_size, d)
            s: vocabulary scores (batch_size, V)
            p: probabilities (batch_size, V)
        """
        # e = x^T E
        e = self.encoder(input_indices)
        # z = e T_r
        z = self.semantic_transition(e, relation)
        # s = z D, p = softmax(s)
        s, p = self.decoder(z)
        return e, z, s, p

    # -------------------------------------------------------------
    # 2. Multiword-to-Next-Word Prediction (Section 10 - 12, 15.3)
    # -------------------------------------------------------------
    def forward_sequence(
        self,
        sequence_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward computation (Section 10, 15.3):
            e_t = x_t^T E
            h_t = tanh([h_{t-1}, e_t] T_R + b)
            z_{t+1} = h_t (via rnn_proj if h != d)
            s = z_{t+1} D
            p = softmax(s)
        Args:
            sequence_indices: tensor of shape (batch_size, seq_len)
        Returns:
            final_h: recurrent state h_t (batch_size, h)
            scores: vocabulary scores (batch_size, V)
            probs: next-word probabilities (batch_size, V)
        """
        # Embed sequence tokens: (batch_size, seq_len, d)
        embeddings = self.encoder(sequence_indices)
        _, final_h = self.rnn_transition(embeddings)
        # Transformed representation z = final_h (Section 15.3, 15.5)
        z = self.rnn_proj(final_h)
        scores, probs = self.decoder(z)
        return final_h, scores, probs

    # -------------------------------------------------------------
    # Loss Computations (Section 5.1, 6.1, 8, 15.1, 15.6)
    # -------------------------------------------------------------
    def compute_semantic_loss(
        self,
        input_indices: torch.Tensor,
        target_indices: torch.Tensor,
        relation: str,
        lambda_syn: float = 0.5,
        lambda_ant: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Total semantic loss (Section 15.1):
            L = L_CE + lambda_syn * L_syn + lambda_ant * L_ant
        Where:
            L_CE = -log p(target)
            L_syn = sum (1 - cos(e_i, e_j))  [for synonyms]
            L_ant = sum (1 + cos(e_i, e_j))  [for antonyms]
        """
        e_input, z, scores, probs = self.forward_semantic(input_indices, relation)
        e_target = self.encoder(target_indices)

        # Cross-entropy prediction loss L_CE = -log p_c
        l_ce = F.cross_entropy(scores, target_indices)

        metrics = {"loss_ce": l_ce.item()}
        total_loss = l_ce

        # Semantic geometric constraints
        cos_sim = F.cosine_similarity(e_input, e_target, dim=-1)

        if relation == "synonym":
            # For synonym: e_i approx e_j -> 1 - cos(e_i, e_j) -> 0 (Section 5.1)
            l_syn = torch.mean(1.0 - cos_sim)
            # Regularize T_syn towards Identity: ||T_syn - I||^2
            T_syn = self.semantic_transition.get_matrix("synonym")
            I = torch.eye(self.embed_dim, device=T_syn.device)
            l_reg = torch.norm(T_syn - I, p="fro") ** 2 / (self.embed_dim ** 2)

            total_loss = total_loss + lambda_syn * (l_syn + 0.1 * l_reg)
            metrics["loss_syn"] = l_syn.item()
            metrics["cos_syn"] = cos_sim.mean().item()

        elif relation == "antonym":
            # For antonym: e_i approx -e_j -> 1 + cos(e_i, e_j) -> 0 (Section 6.1)
            l_ant = torch.mean(1.0 + cos_sim)
            # Regularize T_ant towards -Identity: ||T_ant - (-I)||^2
            T_ant = self.semantic_transition.get_matrix("antonym")
            neg_I = -torch.eye(self.embed_dim, device=T_ant.device)
            l_reg = torch.norm(T_ant - neg_I, p="fro") ** 2 / (self.embed_dim ** 2)

            total_loss = total_loss + lambda_ant * (l_ant + 0.1 * l_reg)
            metrics["loss_ant"] = l_ant.item()
            metrics["cos_ant"] = cos_sim.mean().item()

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics

    def compute_sequence_loss(
        self,
        sequence_indices: torch.Tensor,
        target_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Sequence cross-entropy loss (Section 15.6):
            L_sequence = -log p_{t+1}(w_{t+1})
        """
        _, scores, _ = self.forward_sequence(sequence_indices)
        loss = F.cross_entropy(scores, target_indices)
        return loss, loss.item()

    # -------------------------------------------------------------
    # High-Level Inference Queries
    # -------------------------------------------------------------
    @torch.no_grad()
    def predict_semantic(
        self,
        word_idx: int,
        relation: str,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Runs word -> E -> T_r -> D -> softmax.
        Returns list of (predicted_word_idx, probability).
        """
        x = torch.tensor([word_idx], dtype=torch.long, device=self.encoder.E.device)
        _, _, _, probs = self.forward_semantic(x, relation)
        top_probs, top_indices = torch.topk(probs[0], k=top_k)
        return list(zip(top_indices.tolist(), top_probs.tolist()))

    @torch.no_grad()
    def predict_next(
        self,
        prefix_indices: List[int],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Runs w1...wt -> E -> RNN -> D -> softmax.
        Returns list of (predicted_word_idx, probability).
        """
        seq = torch.tensor([prefix_indices], dtype=torch.long, device=self.encoder.E.device)
        _, _, probs = self.forward_sequence(seq)
        top_probs, top_indices = torch.topk(probs[0], k=top_k)
        return list(zip(top_indices.tolist(), top_probs.tolist()))
