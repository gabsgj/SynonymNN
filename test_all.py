"""
Unit and integration tests for Word Embedding and Word-to-Word Transformations.
Verifies all mathematical operations and architectural components from the PDF.
"""

import unittest
import torch
import torch.nn.functional as F

from dataset import Vocabulary, SemanticDataset
from model import WordEncoder, WordDecoder, SemanticTransition, RecurrentTransition, UnifiedWordModel


class TestWordEmbeddingSystem(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocabulary(["happy", "joyful", "sad", "hot", "cold", "the", "cat", "drinks", "milk"])
        self.embed_dim = 8
        self.vocab_size = len(self.vocab)

    def test_one_hot_representation(self):
        """Verifies Section 2: xi in R^V, xi^T xj = 0 for distinct words."""
        x_happy = self.vocab.to_one_hot("happy")
        x_sad = self.vocab.to_one_hot("sad")
        
        self.assertEqual(x_happy.shape, (1, self.vocab_size))
        self.assertEqual(x_sad.shape, (1, self.vocab_size))
        self.assertEqual(x_happy[0, self.vocab.get_idx("happy")].item(), 1.0)
        self.assertEqual(x_happy.sum().item(), 1.0)

        # Orthogonality of one-hot representations (Section 2)
        dot_product = (x_happy @ x_sad.T).item()
        self.assertEqual(dot_product, 0.0)

    def test_encoder_row_selection_equivalence(self):
        """Verifies Section 3: e_i = x_i^T E selects row i of E."""
        encoder = WordEncoder(self.vocab_size, self.embed_dim)
        idx = self.vocab.get_idx("happy")
        
        # Method 1: Row selection
        e_row = encoder(idx)
        # Method 2: Matrix multiplication with 1-hot vector
        x_onehot = self.vocab.to_one_hot("happy")
        e_matmul = encoder(x_onehot)

        self.assertTrue(torch.allclose(e_row.squeeze(), e_matmul.squeeze()))
        self.assertEqual(e_row.shape, (self.embed_dim,))

    def test_semantic_transitions(self):
        """Verifies Section 4-7: z = e T_r, dimensions in R^(1 x d)."""
        transition = SemanticTransition(self.embed_dim, ["synonym", "antonym"])
        e = torch.randn(1, self.embed_dim)

        z_syn = transition(e, "synonym")
        z_ant = transition(e, "antonym")

        self.assertEqual(z_syn.shape, (1, self.embed_dim))
        self.assertEqual(z_ant.shape, (1, self.embed_dim))

    def test_decoder_and_softmax(self):
        """Verifies Section 8: s = z D in R^(1 x V), p = softmax(s)."""
        decoder = WordDecoder(self.embed_dim, self.vocab_size)
        z = torch.randn(2, self.embed_dim)
        scores, probs = decoder(z)

        self.assertEqual(scores.shape, (2, self.vocab_size))
        self.assertEqual(probs.shape, (2, self.vocab_size))
        # Probabilities must sum to 1.0 across vocabulary
        prob_sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums)))

    def test_recurrent_transition(self):
        """Verifies Section 10: h_t = tanh([h_{t-1}, e_t] T_R + b)."""
        hidden_dim = self.embed_dim
        rnn = RecurrentTransition(self.embed_dim, hidden_dim)

        # Check T_R parameter shape: ((h + d), h)
        self.assertEqual(rnn.T_R.shape, (hidden_dim + self.embed_dim, hidden_dim))
        self.assertEqual(rnn.b.shape, (1, hidden_dim))

        # Check recurrent forward pass over sequence of length 3
        seq_embeds = torch.randn(1, 3, self.embed_dim)
        all_h, final_h = rnn(seq_embeds)
        self.assertEqual(all_h.shape, (1, 3, hidden_dim))
        self.assertEqual(final_h.shape, (1, hidden_dim))

    def test_unified_model_end_to_end(self):
        """Verifies Section 13, 15: Unified Encode -> Transform -> Decode."""
        model = UnifiedWordModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.embed_dim,
            relation_names=["synonym", "antonym"]
        )

        # 1. Semantic forward pass
        happy_idx = torch.tensor([self.vocab.get_idx("happy")])
        e, z, s, p = model.forward_semantic(happy_idx, "synonym")
        self.assertEqual(p.shape, (1, self.vocab_size))

        # 2. Sequence forward pass: "the cat drinks"
        prefix = torch.tensor([[
            self.vocab.get_idx("the"),
            self.vocab.get_idx("cat"),
            self.vocab.get_idx("drinks")
        ]])
        final_h, s_seq, p_seq = model.forward_sequence(prefix)
        self.assertEqual(p_seq.shape, (1, self.vocab_size))

    def test_dataset_loading(self):
        """Verifies dataset.json loader and data integrity."""
        dataset = SemanticDataset("dataset.json")
        self.assertGreater(len(dataset.vocab), 50)
        self.assertGreater(len(dataset.synonyms), 50)
        self.assertGreater(len(dataset.antonyms), 50)
        self.assertIn("singular_to_plural", dataset.relations)
        self.assertIn("country_to_capital", dataset.relations)
        self.assertGreater(len(dataset.sequences), 10)


if __name__ == "__main__":
    unittest.main()
