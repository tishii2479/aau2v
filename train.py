import torch

from analyst import Analyst
from config import parse_config
from dataset import load_dataset_manager


def main() -> None:
    torch.manual_seed(0)

    trainer_config, model_config = parse_config()
    print("trainer_config:", trainer_config)
    print("model_config:", model_config)

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

    def on_epoch_start() -> None:
        analyst.similarity_between_seq_meta_and_item_meta(
            "gender", "M", "genre", method="inner-product", num_top_values=30
        )
        analyst.similarity_between_seq_meta_and_item_meta(
            "gender", "F", "genre", method="inner-product", num_top_values=30
        )

    analyst.fit(on_epoch_start=on_epoch_start, show_fig=False)
    on_epoch_start()


if __name__ == "__main__":
    main()
