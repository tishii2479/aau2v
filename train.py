from analyst import Analyst
from config import parse_args, setup_config
from dataset import load_dataset_manager


def main() -> None:
    args = parse_args()
    trainer_config, model_config = setup_config(args)
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

    def on_epoch_end() -> None:
        analyst.similarity_between_seq_meta_and_item_meta(
            "gender", "M", "genre", method="inner-product", num_top_values=30
        )
        analyst.similarity_between_seq_meta_and_item_meta(
            "gender", "F", "genre", method="inner-product", num_top_values=30
        )

    analyst.fit(on_epoch_end=on_epoch_end, show_fig=False)
    on_epoch_end()


if __name__ == "__main__":
    main()
