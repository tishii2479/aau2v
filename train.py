import os
import pickle

from analyst import Analyst
from config import parse_args, setup_config
from data import SequenceDataset, create_20newsgroup_data, create_hm_data  # noqa


def main() -> None:
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
