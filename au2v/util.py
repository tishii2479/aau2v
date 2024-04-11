import os
import random
from typing import Any, ChainMap, Dict, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def to_full_meta_value(meta_name: str, meta_value: Any) -> str:
    """
    Generate identical string that describes the meta value

    Args:
        meta_name (str): meta data name (column name)
        meta_value (Any): meta data value

    Returns:
        str: identical string that describes the meta value
    """
    return meta_name + ":" + str(meta_value)


def check_model_path(model_path: str) -> None:
    if os.path.exists(model_path):
        response = input(
            f"There is a file at {model_path}, but did not specify `load_model`. "
            + "Is it ok to overwrite? [y/n] "
        )
        if response != "y":
            exit(0)


def get_all_items(
    raw_sequences: Union[Dict[str, List[str]], ChainMap[str, List[str]]]
) -> List[str]:
    st = set()
    for seq in raw_sequences.values():
        for e in seq:
            st.add(e)
    return list(st)


def set_seed(seed: int) -> None:
    # random
    random.seed(seed)

    # numpy
    np.random.seed(seed)

    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def visualize_loss(
    loss_dict: Dict[str, List[float]]
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    fig, ax = plt.subplots()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    for loss_name, losses in loss_dict.items():
        ax.plot(losses, label=loss_name)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def visualize_heatmap(
    data: np.ndarray,
    seq_keys: List[str],
    item_keys: List[str],
    figsize: Tuple[float, float] = (12, 8),
    annot: bool = False,
    cbar: bool = True,
    cmap: str = "OrRd",
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(
        data,
        linewidth=0.2,
        xticklabels=item_keys,
        yticklabels=seq_keys,
        annot=annot,
        cmap=cmap,
        cbar=cbar,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    return fig, ax
