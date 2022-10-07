import os
import pickle
from argparse import ArgumentParser, Namespace
from typing import Tuple

from analyst import Analyst
from config import ModelConfig, TrainerConfig
from data import SequenceDataset, create_20newsgroup_data, create_hm_data  # noqa


def main() -> None:
    def parse_args() -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--num_cluster", type=int, default=10)
        parser.add_argument("--d_model", type=int, default=100)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--epochs", type=int, default=5)
        parser.add_argument("--lr", type=float, default=0.0005)
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--load_model", action="store_true")
        return parser.parse_args()

    def load_dataset(
        dataset_path: str = "data/hm_dataset.pickle",
    ) -> SequenceDataset:
        if os.path.exists(dataset_path):
            print(f"load dataset at: {dataset_path}")
            with open(dataset_path, "rb") as f:
                dataset: SequenceDataset = pickle.load(f)
        else:
            print(f"dataset does not exist at: {dataset_path}, create dataset")
            raw_sequences, item_metadata = create_hm_data(max_data_size=1000)
            dataset = SequenceDataset(
                raw_sequences=raw_sequences,
                item_metadata=item_metadata,
                exclude_metadata_columns=["prod_name"],
            )
            with open(dataset_path, "wb") as f:  # type: ignore
                pickle.dump(dataset, f)
        print("end loading dataset")
        return dataset

    def setup_config(args: Namespace) -> Tuple[TrainerConfig, ModelConfig]:
        trainer_config = TrainerConfig(
            model_name="attentive",
            epochs=args.epochs,
            batch_size=args.batch_size,
            load_model=args.load_model,
            verbose=args.verbose,
            model_path=None,
        )
        model_config = ModelConfig(
            d_model=args.d_model,
            window_size=8,
            negative_sample_size=5,
            lr=args.lr,
            use_learnable_embedding=False,
        )
        return trainer_config, model_config

    args = parse_args()
    trainer_config, model_config = setup_config(args)
    dataset = load_dataset()

    analyst = Analyst(
        dataset=dataset,
        trainer_config=trainer_config,
        model_config=model_config,
    )
    _ = analyst.fit(show_fig=False)

    analyst.top_items(num_cluster=args.num_cluster, show_fig=False)
    _ = analyst.calc_coherence(num_cluster=args.num_cluster)

    analyst.attention_weights_to_meta(0, "colour_group_name")
    analyst.attention_weights_to_sequence(0)


if __name__ == "__main__":
    main()
