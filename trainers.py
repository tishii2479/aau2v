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
from data import SequenceDataset
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

    def attention_weight_to_meta(
        self,
        seq_index: int,
        meta_indicies: List[int],
    ) -> Tensor:
        raise NotImplementedError()

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
    def meta_embedding(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Metadata embedding

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
        dataset: SequenceDataset,
        trainer_config: TrainerConfig,
        model_config: ModelConfig,
    ) -> None:
        self.dataset = dataset
        self.data_loader = DataLoader(dataset, batch_size=trainer_config.batch_size)
        self.trainer_config = trainer_config

        match trainer_config.model_name:
            case "attentive":
                self.model = AttentiveModel(
                    num_seq=dataset.num_seq,
                    num_item=dataset.num_item,
                    num_meta=dataset.num_meta,
                    d_model=model_config.d_model,
                    sequences=dataset.sequences,
                    negative_sample_size=model_config.negative_sample_size,
                )

        self.optimizer = Adam(self.model.parameters(), lr=model_config.lr)

    def _pretrain_embeddings(
        self, dataset: SequenceDataset, d_model: int, items: List[str]
    ) -> Tuple[Tensor, Tensor]:
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
            a = item_embeddings[dataset.item_le.transform(sequence)]
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
            for i, data in enumerate(tqdm.tqdm(self.data_loader)):
                seq_index, item_indicies, meta_indicies, target_index = data

                loss = self.model.forward(
                    seq_index, item_indicies, meta_indicies, target_index
                )
                self.optimizer.zero_grad()
                loss.backward()  # type: ignore
                self.optimizer.step()

                if self.trainer_config.verbose:
                    print(i, len(self.data_loader), loss.item())
                total_loss += loss.item()

            total_loss /= len(self.data_loader)
            if epoch % 1 == 0:
                print(epoch, total_loss)

            losses.append(total_loss)
        print("train end")

        if self.trainer_config.model_path is not None:
            torch.save(self.model.state_dict(), self.trainer_config.model_path)

        if len(losses) > 0:
            print(f"final loss: {losses[-1]}")

        return losses

    @property
    def seq_embedding(self) -> Dict[str, np.ndarray]:
        return {
            seq_name: h_seq.detach().numpy()
            for seq_name, h_seq in zip(
                self.dataset.raw_sequences.keys(), self.model.seq_embedding
            )
        }

    @property
    def item_embedding(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError()

    @property
    def meta_embedding(self) -> Dict[str, Dict[str, np.ndarray]]:
        raise NotImplementedError()


class GensimTrainer(Trainer):
    def __init__(self) -> None:
        raise NotImplementedError()

    def fit(self) -> List[float]:
        raise NotImplementedError()

    @property
    def seq_embedding(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError()

    @property
    def item_embedding(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError()

    @property
    def meta_embedding(self) -> Dict[str, Dict[str, np.ndarray]]:
        raise NotImplementedError()
