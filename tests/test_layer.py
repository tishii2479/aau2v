import unittest

import torch

from layer import (
    EmbeddingDot,
    MetaEmbeddingLayer,
    NegativeSampling,
    WeightSharedNegativeSampling,
    calc_weighted_meta,
)


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

    def test_weighted_meta(self) -> None:
        batch_size = 3
        window_size = 3
        d_model = 2
        item_meta_size = 4

        h_item_meta = torch.rand(batch_size, window_size, item_meta_size, d_model)
        item_meta_weights = torch.rand(batch_size, window_size, item_meta_size)

        h_item_meta_weighted = calc_weighted_meta(h_item_meta, item_meta_weights)

        self.assertTrue(
            h_item_meta_weighted.shape == (batch_size, window_size, d_model)
        )

    def test_weight_shared_negative_sampling(self) -> None:
        batch_size = 3
        d_model = 5
        num_item = 4
        num_item_meta = 5
        num_item_meta_types = 2
        sequences = [
            [1, 2, 0],
            [1, 3, 2],
            [0, 2, 1],
        ]
        item_meta_indicies = torch.LongTensor(
            [
                [0, 2, 4, 0],
                [1, 3, 0, 0],
                [0, 3, 4, 0],
                [1, 2, 0, 0],
            ]
        )
        item_meta_weights = torch.FloatTensor(
            [
                [1, 0.5, 0.5, 0],
                [1, 1, 0, 0],
                [1, 0.5, 0.5, 0],
                [1, 1, 0, 0],
            ]
        )
        embedding_item = MetaEmbeddingLayer(
            num_item,
            num_item_meta,
            num_item_meta_types,
            d_model,
            item_meta_indicies,
            item_meta_weights,
        )
        layer = WeightSharedNegativeSampling(
            d_model=d_model,
            num_item_meta_types=num_item_meta_types,
            sequences=sequences,
            item_meta_indicies=item_meta_indicies,
            item_meta_weights=item_meta_weights,
            embedding_item=embedding_item,
        )
        target_index = torch.LongTensor([0, 0, 1])
        h = torch.rand(batch_size, d_model)

        _ = layer.forward(h, target_index)


if __name__ == "__main__":
    unittest.main()
