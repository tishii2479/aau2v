from typing import Callable, Dict, List, Optional

import torch
import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

from au2v.config import ModelConfig, TrainerConfig
from au2v.dataset_manager import SequenceDatasetManager
from au2v.model import PyTorchModel


class PyTorchTrainer:
    model: PyTorchModel

    def __init__(
        self,
        model: PyTorchModel,
        dataset_manager: SequenceDatasetManager,
        trainer_config: TrainerConfig,
        model_config: ModelConfig,
    ) -> None:
        self.dataset_manager = dataset_manager
        self.data_loaders = {
            "train": DataLoader(
                self.dataset_manager.train_dataset, batch_size=trainer_config.batch_size
            ),
            "valid": DataLoader(
                self.dataset_manager.valid_dataset, batch_size=trainer_config.batch_size
            ),
        }

        self.trainer_config = trainer_config
        self.model_config = model_config
        self.model = model
        self.model.to(self.trainer_config.device)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=model_config.lr,
            weight_decay=model_config.weight_decay,
        )

    def fit(
        self,
        on_train_start: Optional[Callable] = None,
        on_train_end: Optional[Callable] = None,
        on_epoch_start: Optional[Callable[[int], None]] = None,
        on_epoch_end: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, List[float]]:
        if on_train_start is not None:
            on_train_start()

        loss_dict: Dict[str, List[float]] = {name: [] for name in self.data_loaders}
        for epoch in range(self.trainer_config.epochs):
            if on_epoch_start is not None:
                on_epoch_start(epoch)

            total_loss = 0.0
            for data_name, data_loader in self.data_loaders.items():
                match data_name:
                    case "train":
                        self.model.train()
                    case _:
                        self.model.eval()

                for _, data in enumerate(tqdm.tqdm(data_loader)):
                    seq_indices, item_indices, target_indices = data
                    item_indices = torch.stack(item_indices).mT
                    loss = self.model.forward(
                        seq_index=torch.LongTensor(seq_indices).to(
                            self.trainer_config.device
                        ),
                        item_indices=torch.LongTensor(item_indices).to(
                            self.trainer_config.device
                        ),
                        target_index=torch.LongTensor(target_indices).to(
                            self.trainer_config.device
                        ),
                    )
                    loss /= seq_indices.size(0)

                    if data_name == "train":
                        self.optimizer.zero_grad()
                        _ = loss.backward()
                        self.optimizer.step()

                    total_loss += loss.item()

                total_loss /= len(data_loader)
                loss_dict[data_name].append(total_loss)
                print(data_name, total_loss)

            if on_epoch_end is not None:
                on_epoch_end(epoch)

            if self.trainer_config.save_model:
                torch.save(self.model.state_dict(), self.trainer_config.model_path)
                print(f"saved model to {self.trainer_config.model_path}")

        if on_train_end is not None:
            on_train_end()

        return loss_dict
