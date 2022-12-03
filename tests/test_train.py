import unittest

from analyst import Analyst
from config import ModelConfig, TrainerConfig
from dataset import load_dataset_manager


class TestTrain(unittest.TestCase):
    def test_train(self) -> None:
        trainer_config = TrainerConfig(
            model_name="attentive",
            dataset_name="20newsgroup-small",
            epochs=2,
            batch_size=10,
            load_model=False,
            save_model=False,
            load_dataset=False,
            save_dataset=False,
            verbose=False,
            cache_dir="tmp/",
            dataset_dir="tmp/",
        )
        model_config = ModelConfig(
            d_model=50,
            window_size=8,
            negative_sample_size=5,
            dropout=0.1,
            lr=0.0005,
            use_learnable_embedding=False,
            add_seq_embedding=False,
            add_positional_encoding=False,
        )
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

        def on_epoch_end() -> None:
            pass

        analyst.fit(show_fig=False, on_epoch_end=on_epoch_end)


if __name__ == "__main__":
    unittest.main()
