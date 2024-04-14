"""
Example code to train embedding models.
"""

import numpy as np
import torch

from aau2v.config import parse_config
from aau2v.dataset_center import load_dataset_center
from aau2v.model import load_model
from aau2v.trainer import PyTorchTrainer


def main() -> None:
    torch.manual_seed(24)
    np.random.seed(24)

    trainer_config, model_config = parse_config()
    print("trainer_config:", trainer_config)
    print("model_config:", model_config)

    dataset_center = load_dataset_center(
        dataset_name=trainer_config.dataset_name,
        window_size=model_config.window_size,
    )
    model = load_model(
        dataset_center=dataset_center,
        trainer_config=trainer_config,
        model_config=model_config,
    )
    trainer = PyTorchTrainer(
        model=model,
        dataset_center=dataset_center,
        trainer_config=trainer_config,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
