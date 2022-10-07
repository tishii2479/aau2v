import unittest

from analyst import Analyst
from config import default_config
from data import SequenceDataset, create_20newsgroup_data


class TestTrain(unittest.TestCase):
    def test_train(self) -> None:
        raw_sequences, item_metadata = create_20newsgroup_data(
            max_data_size=100, min_seq_length=50
        )
        trainer_config, model_config = default_config()
        dataset = SequenceDataset(
            raw_sequences=raw_sequences, item_metadata=item_metadata
        )
        analyst = Analyst(
            dataset=dataset,
            trainer_config=trainer_config,
            model_config=model_config,
        )
        losses = analyst.fit(show_fig=False)
        self.assertTrue(losses[trainer_config.epochs - 1] <= losses[0])


if __name__ == "__main__":
    unittest.main()
