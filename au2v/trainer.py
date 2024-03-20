import abc
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

from au2v.config import ModelConfig, TrainerConfig
from au2v.dataset_manager import SequenceDatasetManager
from au2v.model import PyTorchModel
from au2v.util import visualize_loss


class Trainer(metaclass=abc.ABCMeta):
    """
    Interface of trainers
    """

    @abc.abstractmethod
    def __init__(
        self,
        dataset_manager: SequenceDatasetManager,
        trainer_config: TrainerConfig,
        model_config: ModelConfig,
    ) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def fit(
        self,
        on_train_start: Optional[Callable] = None,
        on_train_end: Optional[Callable] = None,
        on_epoch_start: Optional[Callable[[int], None]] = None,
        on_epoch_end: Optional[Callable[[int], None]] = None,
        show_fig: bool = False,
    ) -> Dict[str, List[float]]:
        """
        Called to fit to data

        Raises:
            NotImplementedError: if not implemented

        Returns:
            Dict[str, List[float]]: { 損失の名前 : 損失の推移 }
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def eval_loss(self, show_fig: bool = False) -> Tuple[float, Dict[str, float]]:
        """
        予測時の誤差を評価する

        Returns:
            float, Dict[str, float]:
                評価損失, { テストデータ名 : 損失の平均 }
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def eval_pred(self) -> None:
        """
        予測精度を評価する
        """
        raise NotImplementedError()


class PyTorchTrainer(Trainer):
    model: PyTorchModel

    def __init__(
        self,
        model: PyTorchModel,
        dataset_manager: SequenceDatasetManager,
        trainer_config: TrainerConfig,
        model_config: ModelConfig,
    ) -> None:
        self.dataset_manager = dataset_manager
        self.train_data_loader = DataLoader(
            self.dataset_manager.train_dataset, batch_size=trainer_config.batch_size
        )
        if self.dataset_manager.test_datasets is not None:
            self.test_data_loaders: Optional[Dict[str, DataLoader]] = {}
            for test_name, dataset in self.dataset_manager.test_datasets.items():
                test_data_loader = DataLoader(
                    dataset,
                    batch_size=trainer_config.batch_size,
                )
                self.test_data_loaders[test_name] = test_data_loader
        else:
            self.test_data_loaders = None

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
        show_fig: bool = False,
    ) -> Dict[str, List[float]]:
        self.model.train()
        loss_dict: Dict[str, List[float]] = {"train": []}
        best_test_loss = 1e10
        print("train start")

        if on_train_start is not None:
            on_train_start()

        w = self.model_config.window_size

        for epoch in range(self.trainer_config.epochs):
            if on_epoch_start is not None:
                on_epoch_start(epoch)

            total_loss = 0.0
            for i, data in enumerate(tqdm.tqdm(self.train_data_loader)):
                seq_indices, target_indices = data
                item_indices = []
                for seq_index, target_index in zip(seq_indices, target_indices):
                    item_indices.append(
                        (
                            self.dataset_manager.train_dataset.sequences[seq_index][
                                target_index - w : target_index  # noqa
                            ]
                            + self.dataset_manager.train_dataset.sequences[seq_index][
                                target_index + 1 : target_index + w + 1  # noqa
                            ]
                        )
                    )
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
                self.optimizer.zero_grad()
                _ = loss.backward()
                self.optimizer.step()

                if self.trainer_config.verbose:
                    print(i, len(self.train_data_loader), loss.item())
                total_loss += loss.item()

            total_loss /= len(self.train_data_loader)
            loss_dict["train"].append(total_loss)

            if self.test_data_loaders is not None:
                test_loss, test_loss_dict = self.eval_loss(show_fig=False)
                print(
                    f"Epoch: {epoch+1}, loss: {total_loss}, test_loss: {test_loss_dict}"
                )

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    if self.trainer_config.save_model:
                        torch.save(
                            self.model.state_dict(), self.trainer_config.best_model_path
                        )
                        print(
                            f"saved best model to {self.trainer_config.best_model_path}"
                        )

                for loss_name, loss_value in test_loss_dict.items():
                    if loss_name not in loss_dict:
                        loss_dict[loss_name] = []
                    loss_dict[loss_name].append(loss_value)
            else:
                print(f"Epoch: {epoch+1}, loss: {total_loss}")
                if self.trainer_config.save_model:
                    torch.save(
                        self.model.state_dict(), self.trainer_config.best_model_path
                    )
                    print(f"saved best model to {self.trainer_config.best_model_path}")

            if on_epoch_end is not None:
                on_epoch_end(epoch)
        print("train end")

        if on_train_end is not None:
            on_train_end()

        if self.trainer_config.save_model:
            torch.save(self.model.state_dict(), self.trainer_config.model_path)
            print(f"saved model to {self.trainer_config.model_path}")

        if show_fig:
            visualize_loss(loss_dict)

        return loss_dict

    @torch.no_grad()
    def eval_loss(self, show_fig: bool = False) -> Tuple[float, Dict[str, float]]:
        if self.test_data_loaders is None:
            print("No test dataset")
            return 0, {}
        self.model.eval()

        w = self.model_config.window_size
        total_loss_dict: Dict[str, float] = {}

        for test_name, data_loader in self.test_data_loaders.items():
            if self.dataset_manager.test_datasets is None:
                continue
            sequences = self.dataset_manager.test_datasets[test_name].sequences

            pos_outputs: List[float] = []
            neg_outputs: List[float] = []
            total_loss = 0.0
            for i, data in enumerate(tqdm.tqdm(data_loader)):
                seq_indices, target_indices = data
                item_indices = []
                for seq_index, target_index in zip(seq_indices, target_indices):
                    item_indices.append(
                        (
                            sequences[seq_index][
                                target_index - w : target_index  # noqa
                            ]
                            + sequences[seq_index][
                                target_index + 1 : target_index + w + 1  # noqa
                            ]
                        )
                    )

                pos_out, pos_label, neg_out, neg_label = self.model.calc_out(
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

                if show_fig:
                    for e in pos_out.reshape(-1):
                        pos_outputs.append(e.item())
                    for e in neg_out.reshape(-1):
                        neg_outputs.append(e.item())

                loss_pos = F.binary_cross_entropy(pos_out, pos_label)
                loss_neg = F.binary_cross_entropy(neg_out, neg_label)
                loss = (
                    loss_pos + loss_neg / self.model_config.negative_sample_size
                ) / 2

                loss /= seq_indices.size(0)
                total_loss += loss.item()

            total_loss /= len(data_loader)
            total_loss_dict[test_name] = total_loss

            if show_fig:
                plt.hist(pos_outputs)
                plt.show()
                plt.hist(neg_outputs)
                plt.show()

        eval_loss = 0.0
        for val_loss in total_loss_dict.values():
            eval_loss += val_loss
        eval_loss /= len(total_loss_dict)

        return eval_loss, total_loss_dict

    @torch.no_grad()
    def eval_pred(self) -> None:
        if self.test_data_loaders is None:
            print("No test dataset")
            return
        self.model.eval()
        output_embeddings = self.model.output_item_embedding.detach().numpy()
        correct_count = 0
        total_count = 0
        for test_name, data_loader in self.test_data_loaders.items():
            for i, data in enumerate(tqdm.tqdm(data_loader)):
                (
                    seq_index,
                    item_indices,
                    target_index,
                ) = data

                p = self.model.calc_prediction_vector(
                    seq_index=seq_index,
                    item_indices=item_indices,
                )
                p = p.detach().numpy()
                v = np.dot(p, output_embeddings.T)
                rank = v.argsort()
                pred = rank[:, 0:100]
                target = target_index.detach().numpy()
                for p, t in zip(pred, target):
                    if t in p:
                        print(f"{correct_count}, {total_count}")
                        correct_count += 1
                    total_count += 1

        print(f"accuracy: {correct_count / total_count}")
