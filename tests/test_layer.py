import unittest

import torch

import aau2v.layer


class TestLayer(unittest.TestCase):
    def test_embedding_dot(self) -> None:
        lay = aau2v.layer.EmbeddingDot(d_model=3, num_item=4)
        embedding_weight = lay.embedding.weight
        h = torch.Tensor([[[0, 1, 0]], [[1, 0, 0]]])
        indices = torch.tensor([[1], [2]], dtype=torch.long)
        out = lay.forward(h, indices)
        self.assertAlmostEqual(out[0][0][0], embedding_weight[1][1])
        self.assertAlmostEqual(out[1][0][0], embedding_weight[2][0])

        indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out = lay.forward(h, indices)
        self.assertAlmostEqual(out[0][0][0], embedding_weight[0][1])
        self.assertAlmostEqual(out[0][0][1], embedding_weight[1][1])
        self.assertAlmostEqual(out[1][0][0], embedding_weight[1][0])
        self.assertAlmostEqual(out[1][0][1], embedding_weight[2][0])

    def test_weighted_meta(self) -> None:
        batch_size = 3
        window_size = 3
        d_model = 2
        item_meta_size = 4

        h_item_meta = torch.rand(batch_size, window_size, item_meta_size, d_model)
        item_meta_weights = torch.rand(batch_size, window_size, item_meta_size)

        h_item_meta_weighted = aau2v.layer.calc_weighted_meta(
            h_item_meta, item_meta_weights
        )

        self.assertTrue(
            h_item_meta_weighted.shape == (batch_size, window_size, d_model)
        )


if __name__ == "__main__":
    unittest.main()
