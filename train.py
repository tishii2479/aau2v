import os
import pickle

from analyst import Analyst
from config import parse_args, setup_config
from data import SequenceDatasetManager, create_movielens_data, create_toydata  # noqa


def load_dataset(
    dataset_path: str,
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
            test_raw_sequences_dict,
        ) = create_movielens_data()
        dataset_manager = SequenceDatasetManager(
            train_raw_sequences=train_raw_sequences,
            test_raw_sequences_dict=test_raw_sequences_dict,
            item_metadata=item_metadata,
            seq_metadata=seq_metadata,
            exclude_item_metadata_columns=["title"],
        )
        with open(dataset_path, "wb") as f:
            pickle.dump(dataset_manager, f)
    print("end loading dataset")
    return dataset_manager


def main() -> None:
    args = parse_args()
    trainer_config, model_config = setup_config(args)
    print("trainer_config:", trainer_config)
    print("model_config:", model_config)

    dataset_manager = load_dataset(trainer_config.dataset_path)

    analyst = Analyst(
        dataset_manager=dataset_manager,
        trainer_config=trainer_config,
        model_config=model_config,
    )
    analyst.fit(show_fig=False)

    analyst.eval_prediction_accuracy()

    # _, loss_dict = analyst.eval_prediction_loss()
    # print("loss_dict:", loss_dict)

    # analyst.top_items(num_cluster=args.num_cluster, show_fig=False)
    # _ = analyst.calc_coherence(num_cluster=args.num_cluster)

    # analyst.similarity_between_seq_and_item(0)

    # analyst.cluster_embeddings(args.num_cluster)

    # analyst.similar_items(0)
    # analyst.similar_sequences(0)

    # analyst.similarity_between_seq_meta_and_item_meta(
    #     "age", "18-24", "genre", method="inner-product", num_top_values=30
    # )
    # analyst.similarity_between_seq_meta_and_item_meta(
    #     "gender", "F", "genre", method="inner-product", num_top_values=30
    # )
    # analyst.similarity_between_seq_meta_and_item_meta(
    #     "gender", "M", "genre", method="attention", num_top_values=30
    # )
    # analyst.similarity_between_seq_meta_and_item_meta(
    #     "gender", "F", "genre", method="attention", num_top_values=30
    # )

    # analyst.visualize_meta_embedding("gender", "genre")


if __name__ == "__main__":
    main()
