import abc
from typing import Dict, List

import numpy as np

from config import ModelConfig, TrainerConfig
from data import Sequence, SequenceDataset
from models import AttentiveModel


class Trainer(metaclass=abc.ABCMeta):
    """
    Interface of trainers
    """

    @abc.abstractmethod
    def __init__(
        self,
        raw_sequences: List[Sequence],
        trainer_config: TrainerConfig,
        model_config: ModelConfig,
    ) -> None:
        raise NotImplementedError()

    def fit(self) -> List[float]:
        """
        Called to fit to data

        Raises:
            NotImplementedError: if not implemented

        Returns:
            List[float]: losses
        """
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
    def __init__(
        self,
        dataset: SequenceDataset,
        trainer_config: TrainerConfig,
        model_config: ModelConfig,
    ) -> None:
        self.dataset = dataset
        self.trainer_config = trainer_config

        match trainer_config.model_name:
            case "attentive":
                self.model = AttentiveModel(
                    num_seq=self.dataset.num_seq,
                    num_item=self.dataset.num_item,
                    d_model=model_config.d_model,
                    sequences=self.dataset.sequences,
                    negative_sample_size=model_config.negative_sample_size,
                )

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
