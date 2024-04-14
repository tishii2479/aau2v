import unittest

from aau2v.config import ModelConfig, TrainerConfig
from aau2v.dataset_center import load_dataset_center
from aau2v.model import load_model
from aau2v.trainer import PyTorchTrainer


class TestTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer_config = TrainerConfig(
            epochs=2,
            save_model=False,
        )
        self.model_config = ModelConfig(window_size=1)

    def test_train(self) -> None:
        model_names = ["aau2v", "user2vec"]
        dataset_center = load_dataset_center(
            dataset_name=self.trainer_config.dataset_name,
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
            )
            trainer.fit()


if __name__ == "__main__":
    unittest.main()
