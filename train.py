import os
import pickle

from analyst import Analyst
from config import parse_args, setup_config
from data import SequenceDatasetManager, create_hm_data


def main() -> None:
    def load_dataset(
        dataset_path: str = "data/hm_dataset.pickle",
    ) -> SequenceDatasetManager:
        if os.path.exists(dataset_path):
            print(f"load dataset at: {dataset_path}")
            with open(dataset_path, "rb") as f:
                dataset_manager: SequenceDatasetManager = pickle.load(f)
        else:
            print(f"dataset does not exist at: {dataset_path}, create dataset")
            (
                train_raw_sequences,
                item_metadata,
                seq_metadata,
                test_raw_sequences,
            ) = create_hm_data(max_data_size=1000, test_data_size=500)
            dataset_manager = SequenceDatasetManager(
                train_raw_sequences=train_raw_sequences,
                test_raw_sequences=test_raw_sequences,
                item_metadata=item_metadata,
                seq_metadata=seq_metadata,
                exclude_item_metadata_columns=["prod_name"],
            )
            with open(dataset_path, "wb") as f:
                pickle.dump(dataset_manager, f)
        print("end loading dataset")
        return dataset_manager

    args = parse_args()
    trainer_config, model_config = setup_config(args)
    trainer_config.model_path = "weights/model_hm_meta.pt"
    print("trainer_config:", trainer_config)
    print("model_config:", model_config)

    dataset_manager = load_dataset("data/hm_dataset_with_test.pickle")

    analyst = Analyst(
        dataset_manager=dataset_manager,
        trainer_config=trainer_config,
        model_config=model_config,
    )
    _ = analyst.fit(show_fig=False)

    analyst.prediction_accuracy()

    # analyst.top_items(num_cluster=args.num_cluster, show_fig=False)
    # _ = analyst.calc_coherence(num_cluster=args.num_cluster)

    # analyst.attention_weights_to_meta(0, "colour_group_name")
    # analyst.attention_weights_to_sequence(0)

    # analyst.cluster_embeddings(args.num_cluster)
    # analyst.similar_items(0)


if __name__ == "__main__":
    main()
