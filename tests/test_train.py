import unittest

from analyst import Analyst
from config import ModelConfig, TrainerConfig
from dataset import load_dataset_manager


class TestTrain(unittest.TestCase):
    def test_train(self) -> None:
        trainer_config = TrainerConfig(
            epochs=2,
            load_model=False,
            save_model=False,
            load_dataset=False,
            save_dataset=False,
            cache_dir="tmp/",
            dataset_dir="tmp/",
        )
        model_config = ModelConfig()
        dataset_manager = load_dataset_manager(
            dataset_name=trainer_config.dataset_name,
            dataset_dir=trainer_config.dataset_dir,
            load_dataset=trainer_config.load_dataset,
            save_dataset=trainer_config.save_dataset,
        )
        analyst = Analyst(
            dataset_manager=dataset_manager,
            trainer_config=trainer_config,
            model_config=model_config,
        )

        analyst.fit(show_fig=False)


if __name__ == "__main__":
    unittest.main()
