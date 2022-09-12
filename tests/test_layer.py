import unittest

import torch
from layer import EmbeddingDot, NegativeSampling, PositionalEncoding


class TestLayer(unittest.TestCase):
    def test_negative_sampling(self) -> None:
        sequences = [
            [1, 2, 0],
            [1, 3, 2],
            [0, 2, 1],
        ]
        layer = NegativeSampling(
            d_model=3, num_item=4, sequences=sequences, power=1, negative_sample_size=5
        )
        h = torch.Tensor([[0, 1, 0]])
        target_index = torch.tensor([0], dtype=torch.long)
        _ = layer.forward(h, target_index)
        self.assertAlmostEqual(1, layer.sampler.word_p.sum())

    def test_embedding_dot(self) -> None:
        layer = EmbeddingDot(d_model=3, num_item=4)
        embedding_weight = layer.embedding.weight
        h = torch.Tensor([[[0, 1, 0]], [[1, 0, 0]]])
        indicies = torch.tensor([[1], [2]], dtype=torch.long)
        out = layer.forward(h, indicies)
        self.assertAlmostEqual(out[0][0][0], embedding_weight[1][1])
        self.assertAlmostEqual(out[1][0][0], embedding_weight[2][0])

        indicies = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out = layer.forward(h, indicies)
        self.assertAlmostEqual(out[0][0][0], embedding_weight[0][1])
        self.assertAlmostEqual(out[0][0][1], embedding_weight[1][1])
        self.assertAlmostEqual(out[1][0][0], embedding_weight[1][0])
        self.assertAlmostEqual(out[1][0][1], embedding_weight[2][0])

    def test_positional_encoding(self) -> None:
        batch_size = 2
        d_model = 4
        seq_length = 3
        layer = PositionalEncoding(
            d_model=d_model, max_sequence_length=seq_length, dropout=0.1
        )
        hs = torch.rand(batch_size, seq_length, d_model)
        _ = layer.forward(hs)


if __name__ == "__main__":
    unittest.main()
