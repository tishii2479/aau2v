import unittest

from analyst import Analyst
from config import default_config
from data import SequenceDatasetManager, create_20newsgroup_data


class TestTrain(unittest.TestCase):
    def test_train(self) -> None:
        trainer_config, model_config = default_config()
        (
            train_raw_sequences,
            item_metadata,
            seq_metadata,
            test_raw_sequences,
        ) = create_20newsgroup_data(max_data_size=10, test_data_size=50)
        dataset_manager = SequenceDatasetManager(
            train_raw_sequences=train_raw_sequences,
            item_metadata=item_metadata,
            seq_metadata=seq_metadata,
            test_raw_sequences=test_raw_sequences,
            exclude_item_metadata_columns=["prod_name"],
        )
        analyst = Analyst(
            dataset_manager=dataset_manager,
            trainer_config=trainer_config,
            model_config=model_config,
        )
        losses = analyst.fit(show_fig=False)
        self.assertTrue(losses[trainer_config.epochs - 1] <= losses[0])


if __name__ == "__main__":
    unittest.main()
