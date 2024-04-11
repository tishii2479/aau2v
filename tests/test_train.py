import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))

from aau2v.config import ModelConfig, TrainerConfig  # noqa
from aau2v.dataset_center import load_dataset_center  # noqa
from aau2v.model import load_model  # noqa
from aau2v.trainer import PyTorchTrainer  # noqa


class TestTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer_config = TrainerConfig(
            epochs=2,
            save_model=False,
            load_dataset=False,
            save_dataset=False,
            ignore_saved_model=True,
            model_dir="tmp/",
            dataset_dir="tmp/",
        )
        self.model_config = ModelConfig(window_size=1)

    def test_train(self) -> None:
        model_names = ["aau2v", "user2vec"]
        dataset_center = load_dataset_center(
            dataset_name=self.trainer_config.dataset_name,
            dataset_dir=self.trainer_config.dataset_dir,
            load_dataset=self.trainer_config.load_dataset,
            save_dataset=self.trainer_config.save_dataset,
            window_size=self.model_config.window_size,
        )
        for model_name in model_names:
            self.trainer_config.model_name = model_name
            model = load_model(
                dataset_center=dataset_center,
                trainer_config=self.trainer_config,
                model_config=self.model_config,
            )
            trainer = PyTorchTrainer(
                model=model,
                dataset_center=dataset_center,
                trainer_config=self.trainer_config,
                model_config=self.model_config,
            )
            trainer.fit()


if __name__ == "__main__":
    unittest.main()
