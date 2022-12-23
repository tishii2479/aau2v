import unittest

from analyst import Analyst
from config import ModelConfig, TrainerConfig
from dataset import load_dataset_manager
from trainers import PyTorchTrainer


class TestTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer_config = TrainerConfig(
            epochs=2,
            load_model=False,
            save_model=False,
            load_dataset=False,
            save_dataset=False,
            model_dir="tmp/",
            dataset_dir="tmp/",
        )
        self.model_config = ModelConfig()

    def test_train(self) -> None:
        models = ["attentive2", "attentive", "doc2vec"]
        dataset_manager = load_dataset_manager(
            dataset_name=self.trainer_config.dataset_name,
            dataset_dir=self.trainer_config.dataset_dir,
            load_dataset=self.trainer_config.load_dataset,
            save_dataset=self.trainer_config.save_dataset,
        )
        for model in models:
            self.trainer_config.model_name = model
            trainer = PyTorchTrainer(
                dataset_manager=dataset_manager,
                trainer_config=self.trainer_config,
                model_config=self.model_config,
            )
            trainer.fit(show_fig=False)

    def test_analyst(self) -> None:
        dataset_manager = load_dataset_manager(
            dataset_name=self.trainer_config.dataset_name,
            dataset_dir=self.trainer_config.dataset_dir,
            load_dataset=self.trainer_config.load_dataset,
            save_dataset=self.trainer_config.save_dataset,
        )
        trainer = PyTorchTrainer(
            dataset_manager=dataset_manager,
            trainer_config=self.trainer_config,
            model_config=self.model_config,
        )
        analyst = Analyst(trainer.model, dataset_manager)
        print(analyst.similarity_between_seq_and_item(0))
        print(analyst.analyze_seq(0))


if __name__ == "__main__":
    unittest.main()
