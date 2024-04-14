from pathlib import Path
from typing import Callable, Optional

import torch
import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

from aau2v.config import TrainerConfig
from aau2v.dataset_center import SequenceDatasetCenter
from aau2v.model import PyTorchModel


class PyTorchTrainer:
    model: PyTorchModel

    def __init__(
        self,
        model: PyTorchModel,
        dataset_center: SequenceDatasetCenter,
        trainer_config: TrainerConfig,
    ) -> None:
        self.data_loaders = {
            "train": DataLoader(
                dataset_center.train_dataset, batch_size=trainer_config.batch_size
            ),
            "valid": DataLoader(
                dataset_center.valid_dataset, batch_size=trainer_config.batch_size
            ),
        }

        self.config = trainer_config
        self.model = model
        self.model.to(self.config.device)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=trainer_config.lr,
            weight_decay=trainer_config.weight_decay,
        )

    def fit(
        self,
        on_epoch_end: Optional[Callable[[int], None]] = None,
        on_iter_end: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, list[float]]:
        loss_dict: dict[str, list[float]] = {name: [] for name in self.data_loaders}
        for epoch in range(self.config.epochs):

            total_loss = 0.0
            for data_name, data_loader in self.data_loaders.items():
                match data_name:
                    case "train":
                        self.model.train()
                    case _:
                        self.model.eval()

                for batch, data in enumerate(tqdm.tqdm(data_loader)):
                    seq_indices, item_indices, target_indices = data
                    item_indices = torch.stack(item_indices).mT
                    loss = self.model.forward(
                        seq_index=torch.LongTensor(seq_indices).to(self.config.device),
                        item_indices=torch.LongTensor(item_indices).to(
                            self.config.device
                        ),
                        target_index=torch.LongTensor(target_indices).to(
                            self.config.device
                        ),
                    )
                    loss /= seq_indices.size(0)

                    if data_name == "train":
                        self.optimizer.zero_grad()
                        _ = loss.backward()  # type: ignore
                        self.optimizer.step()

                    total_loss += loss.item()

                    if on_iter_end is not None:
                        on_iter_end(epoch, batch)

                total_loss /= len(data_loader)
                loss_dict[data_name].append(total_loss)
                print(f"loss for {data_name}: {total_loss:.8}")

            if on_epoch_end is not None:
                on_epoch_end(epoch)

            if self.config.save_model:
                model_path = (
                    Path(self.config.model_dir) / f"{self.config.model_name}-{epoch}.pt"
                )
                torch.save(self.model, model_path)
                print(f"saved model to {model_path}")

        return loss_dict
