import unittest

from analyst import Analyst
from config import ModelConfig, TrainerConfig
from dataset import load_dataset_manager
from trainers import PyTorchTrainer


class TestTrain(unittest.TestCase):
    def test_train(self) -> None:
        trainer_config = TrainerConfig(
            epochs=2,
            dataset_name="toydata-small",
            load_model=False,
            save_model=False,
            load_dataset=False,
            save_dataset=False,
            model_dir="tmp/",
            dataset_dir="tmp/",
        )
        model_config = ModelConfig()
        dataset_manager = load_dataset_manager(
            dataset_name=trainer_config.dataset_name,
            dataset_dir=trainer_config.dataset_dir,
            load_dataset=trainer_config.load_dataset,
            save_dataset=trainer_config.save_dataset,
        )
        trainer = PyTorchTrainer(
            dataset_manager=dataset_manager,
            trainer_config=trainer_config,
            model_config=model_config,
        )
        analyst = Analyst(trainer.model, dataset_manager)

        trainer.fit(show_fig=False)

        print(analyst.similarity_between_seq_and_item_meta(0, "genre"))
        print(analyst.similarity_between_seq_and_item(0))
        print(analyst.similarity_between_seq_meta_and_item_meta("gender", "M", "genre"))
        print(analyst.analyze_seq(0))


if __name__ == "__main__":
    unittest.main()
