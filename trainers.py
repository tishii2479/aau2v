import abc
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import tqdm
from gensim.models import word2vec
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import ModelConfig, TrainerConfig
from data import SequenceDataset, SequenceDatasetManager
from model import AttentiveModel, PyTorchModel


class Trainer(metaclass=abc.ABCMeta):
    """
    Interface of trainers
    """

    @abc.abstractmethod
    def __init__(
        self,
        raw_sequences: List[Tuple[str, Dict[str, Any]]],
        trainer_config: TrainerConfig,
        model_config: ModelConfig,
    ) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def fit(self) -> List[float]:
        """
        Called to fit to data

        Raises:
            NotImplementedError: if not implemented

        Returns:
            List[float]: losses
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def eval(self) -> float:
        """
        予測精度を評価する

        Returns:
            float: 予測精度
        """
        raise NotImplementedError()

    @abc.abstractmethod
    @torch.no_grad()  # type: ignore
    def attention_weight_to_meta(
        self,
        seq_index: int,
        meta_indicies: List[int],
    ) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    @torch.no_grad()  # type: ignore
    def attention_weight_to_item(
        self, seq_index: int, item_indicies: List[int]
    ) -> Tensor:
        raise NotImplementedError()

    @abc.abstractproperty
    @property
    def seq_embedding(self) -> Dict[str, np.ndarray]:
        """
        Sequence embedding

        Raises:
            NotImplementedError: if not implemented

        Returns:
            Dict[str, np.ndarray]: (sequence id, embedding)
        """
        raise NotImplementedError()

    @abc.abstractproperty
    @property
    def item_embedding(self) -> Dict[str, np.ndarray]:
        """
        Item embedding

        Raises:
            NotImplementedError: if not implemented

        Returns:
            Dict[str, np.ndarray]: (item id, embedding)
        """
        raise NotImplementedError()

    @abc.abstractproperty
    @property
    def item_meta_embedding(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Item Metadata embedding

        Raises:
            NotImplementedError: if not implemented

        Returns:
            Dict[str, Dict[str, np.ndarray]]: (meta_name, (meta_value, embedding))
        """
        raise NotImplementedError()

    @abc.abstractproperty
    @property
    def seq_meta_embedding(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Sequence Metadata embedding

        Raises:
            NotImplementedError: if not implemented

        Returns:
            Dict[str, Dict[str, np.ndarray]]: (meta_name, (meta_value, embedding))
        """
        raise NotImplementedError()


class PyTorchTrainer(Trainer):
    model: PyTorchModel

    def __init__(
        self,
        dataset_manager: SequenceDatasetManager,
        trainer_config: TrainerConfig,
        model_config: ModelConfig,
    ) -> None:
        self.dataset_manager = dataset_manager
        self.train_data_loader = DataLoader(
            self.dataset_manager.train_dataset, batch_size=trainer_config.batch_size
        )
        if self.dataset_manager.test_dataset is not None:
            self.test_data_loader = DataLoader(
                self.dataset_manager.test_dataset, batch_size=trainer_config.batch_size
            )
        self.trainer_config = trainer_config

        match trainer_config.model_name:
            case "attentive":
                self.model = AttentiveModel(
                    num_seq=self.dataset_manager.num_seq,
                    num_item=self.dataset_manager.num_item,
                    num_seq_meta=dataset_manager.num_seq_meta,
                    num_item_meta=self.dataset_manager.num_item_meta,
                    d_model=model_config.d_model,
                    sequences=self.dataset_manager.sequences,
                    negative_sample_size=model_config.negative_sample_size,
                )

        if (
            self.trainer_config.load_model
            and self.trainer_config.model_path is not None
        ):
            print(f"load_state_dict from: {self.trainer_config.model_path}")
            loaded = torch.load(self.trainer_config.model_path)  # type: ignore
            self.model.load_state_dict(loaded)

        self.optimizer = Adam(self.model.parameters(), lr=model_config.lr)

    def _pretrain_embeddings(
        self, dataset: SequenceDataset, d_model: int, items: List[str]
    ) -> Tuple[Tensor, Tensor]:
        # TODO: オプションを付ける
        print("word2vec start.")
        word2vec_model = word2vec.Word2Vec(
            sentences=dataset.raw_sequences, vector_size=d_model, min_count=1
        )
        print("word2vec end.")
        item_embeddings = torch.Tensor(
            [list(word2vec_model.wv[item]) for item in items]
        )

        print("learn_sequence_embedding start")

        # TODO: refactor
        seq_embedding_list = []
        for sequence in tqdm.tqdm(dataset.raw_sequences):
            a = item_embeddings[self.dataset_manager.item_le.transform(sequence)]
            seq_embedding_list.append(list(a.mean(dim=0)))

        seq_embeddings = torch.Tensor(seq_embedding_list)
        print("learn_sequence_embedding end")

        return item_embeddings, seq_embeddings

    def fit(self) -> List[float]:
        self.model.train()
        losses = []
        print("train start")
        for epoch in range(self.trainer_config.epochs):
            total_loss = 0.0
            for i, data in enumerate(tqdm.tqdm(self.train_data_loader)):
                (
                    seq_index,
                    item_indicies,
                    seq_meta_indicies,
                    item_meta_indicies,
                    target_index,
                ) = data

                loss = self.model.forward(
                    seq_index=seq_index,
                    item_indicies=item_indicies,
                    seq_meta_indicies=seq_meta_indicies,
                    item_meta_indicies=item_meta_indicies,
                    target_index=target_index,
                )
                self.optimizer.zero_grad()
                loss.backward()  # type: ignore
                self.optimizer.step()

                if self.trainer_config.verbose:
                    print(i, len(self.train_data_loader), loss.item())
                total_loss += loss.item()

            total_loss /= len(self.train_data_loader)
            if epoch % 1 == 0:
                print(epoch, total_loss)

            losses.append(total_loss)
        print("train end")

        if self.trainer_config.model_path is not None:
            torch.save(self.model.state_dict(), self.trainer_config.model_path)

        if len(losses) > 0:
            print(f"final loss: {losses[-1]}")

        return losses

    @torch.no_grad()
    def eval(self) -> float:
        self.model.eval()
        pos_outputs: List[float] = []
        neg_outputs: List[float] = []
        for i, data in enumerate(tqdm.tqdm(self.test_data_loader)):
            (
                seq_index,
                item_indicies,
                item_meta_indicies,
                seq_meta_indicies,
                target_index,
            ) = data

            pos_out, pos_label, neg_out, neg_label = self.model.calc_out(
                seq_index,
                item_indicies,
                item_meta_indicies,
                seq_meta_indicies,
                target_index,
            )
            for e in pos_out.reshape(-1):
                pos_outputs.append(e.item())
            for e in neg_out.reshape(-1):
                neg_outputs.append(e.item())

        import matplotlib.pyplot as plt

        plt.hist(pos_outputs)
        plt.show()
        plt.hist(neg_outputs)
        plt.show()
        return 0

    def attention_weight_to_item(
        self, seq_index: int, item_indicies: List[int]
    ) -> Tensor:
        return self.model.attention_weight_to_item(seq_index, item_indicies)

    def attention_weight_to_meta(
        self, seq_index: int, meta_indicies: List[int]
    ) -> Tensor:
        return self.model.attention_weight_to_meta(seq_index, meta_indicies)

    @property
    def seq_embedding(self) -> Dict[str, np.ndarray]:
        return {
            seq_name: h_seq.detach().numpy()
            for seq_name, h_seq in zip(
                self.dataset_manager.seq_le.classes_, self.model.seq_embedding
            )
        }

    @property
    def item_embedding(self) -> Dict[str, np.ndarray]:
        return {
            item_name: h_item.detach().numpy()
            for item_name, h_item in zip(
                self.dataset_manager.item_le.classes_, self.model.item_embedding
            )
        }

    @property
    def seq_meta_embedding(self) -> Dict[str, Dict[str, np.ndarray]]:
        raise NotImplementedError()

    @property
    def item_meta_embedding(self) -> Dict[str, Dict[str, np.ndarray]]:
        raise NotImplementedError()


class GensimTrainer(Trainer):
    def __init__(self) -> None:
        raise NotImplementedError()

    def fit(self) -> List[float]:
        raise NotImplementedError()

    def eval(self) -> float:
        raise NotImplementedError()

    @property
    def seq_embedding(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError()

    @property
    def item_embedding(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError()

    @property
    def seq_meta_embedding(self) -> Dict[str, Dict[str, np.ndarray]]:
        raise NotImplementedError()

    @property
    def item_meta_embedding(self) -> Dict[str, Dict[str, np.ndarray]]:
        raise NotImplementedError()
