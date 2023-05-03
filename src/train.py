import numpy as np
import torch

from config import parse_config
from dataset_manager import load_dataset_manager
from model import load_model
from trainers import PyTorchTrainer


def main() -> None:
    torch.manual_seed(24)
    np.random.seed(24)

    trainer_config, model_config = parse_config()
    print("trainer_config:", trainer_config)
    print("model_config:", model_config)

    dataset_manager = load_dataset_manager(
        dataset_name=trainer_config.dataset_name,
        dataset_dir=trainer_config.dataset_dir,
        load_dataset=trainer_config.load_dataset,
        save_dataset=trainer_config.save_dataset,
        window_size=model_config.window_size,
    )
    model = load_model(
        dataset_manager=dataset_manager,
        trainer_config=trainer_config,
        model_config=model_config,
    )
    trainer = PyTorchTrainer(
        model=model,
        dataset_manager=dataset_manager,
        trainer_config=trainer_config,
        model_config=model_config,
    )
    trainer.fit(show_fig=False)


if __name__ == "__main__":
    main()
