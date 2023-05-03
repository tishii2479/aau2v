import os
from math import log
from typing import Any, ChainMap, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.manifold import TSNE  # noqa


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


def visualize_cluster(
    features: List[np.ndarray],
    num_cluster: int,
    cluster_labels: List[int],
    answer_labels: Optional[List[int]] = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    r"""
    Visualize cluster to 2d
    """
    pca = PCA(n_components=2, random_state=0)
    pca.fit(features)
    pca_features = pca.fit_transform(features)

    colors = cm.rainbow(np.linspace(0, 1, num_cluster))

    fig, ax = plt.subplots()

    for i in range(pca_features.shape[0]):
        if answer_labels is not None:
            marker = "$" + str(answer_labels[i]) + "$"
        else:
            marker = "."
        ax.scatter(
            x=pca_features[i, 0],
            y=pca_features[i, 1],
            color=colors[cluster_labels[i]],
            marker=marker,
        )
    return fig, ax


def visualize_vectors(
    embeddings: Dict[str, np.ndarray], method: str = "pca"
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    # TODO: return ax
    vector_names = np.array(list(embeddings.keys()))
    vector_values = np.array(list(embeddings.values()))

    match method:
        case "pca":
            dec = PCA(n_components=2, random_state=0)
        case "tsne":
            dec = TSNE(
                n_components=2, random_state=0, learning_rate="auto", init="random"
            )
        case _:
            assert False, f"Invalid method {method}"

    dec.fit(vector_values)
    dec_features = dec.fit_transform(vector_values)

    fig, ax = plt.subplots()

    for i in range(dec_features.shape[0]):
        marker = "."
        norm_vec = dec_features[i]
        ax.scatter(
            x=norm_vec[0],
            y=norm_vec[1],
            marker=marker,
        )
        ax.annotate(vector_names[i], (norm_vec[0], norm_vec[1]))
    fig.savefig("data/vis_vectors.svg", format="svg")
    return fig, ax


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


def top_cluster_items(
    num_cluster: int,
    cluster_occurence_array: np.ndarray,
    cluster_size: List[int],
    num_top_item: int,
) -> List[Tuple[List[int], List[float]]]:
    r"""
    Args:
        num_cluster: number of clusters
        topic_indices: list of sequence indices for each topic
            shape: (num_topic, len(sequence_indices))
        sequences: all sequence data
        num_top_item: number of item to list
        num_item: number of items in data
    Return:
        top items for each cluster
            shape: (num_topic, num_top_item, (item_list, item_ratio))
    """
    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(cluster_occurence_array).toarray()

    top_items = []
    for cluster in range(num_cluster):
        # Get item index of top items which has largest item_count
        top_items_for_cluster = list(tf_idf[cluster].argsort()[::-1][:num_top_item])
        top_item_counts = np.sort(tf_idf[cluster])[::-1][:num_top_item]
        top_items_for_cluster_counts = list(top_item_counts / cluster_size[cluster])
        top_items.append((top_items_for_cluster, top_items_for_cluster_counts))
    return top_items


def check_model_path(model_path: str) -> None:
    if os.path.exists(model_path):
        response = input(
            f"There is a file at {model_path}, but did not specify `load_model`. "
            + "Is it ok to overwrite? [y/n] "
        )
        if response != "y":
            exit(0)


def calc_cluster_occurence_array(
    num_cluster: int,
    cluster_labels: List[int],
    sequences: List[List[int]],
    num_item: int,
) -> Tuple[np.ndarray, List[int]]:
    occurence_array = np.zeros((num_cluster, num_item))
    cluster_size = [0] * num_cluster
    for i, sequence in enumerate(sequences):
        cluster_size[cluster_labels[i]] += 1
        for item_index in set(sequence):
            occurence_array[cluster_labels[i]][item_index] += 1
    return occurence_array, cluster_size


def calc_sequence_occurence_array(
    sequences: List[List[int]], num_item: int
) -> np.ndarray:
    occurence_array = np.zeros((len(sequences), num_item))
    for i, sequence in enumerate(sequences):
        for item_index in set(sequence):
            occurence_array[i][item_index] += 1
    return occurence_array


def calc_coherence(
    sequence_occurence_array: np.ndarray,
    top_item_infos: List[Tuple[List[int], List[float]]],
) -> float:
    coherence_sum = 0.0
    for cluster, (top_items, _) in enumerate(top_item_infos):
        coherence = 0.0
        for i in top_items:
            for j in top_items:
                # ignore duplicate pairs
                if i <= j:
                    continue
                d_ij = (sequence_occurence_array[:, i] > 0) & (
                    sequence_occurence_array[:, j] > 0
                )
                u_ij = d_ij.sum()
                d_i = sequence_occurence_array[:, i]
                u_i = d_i.sum()
                coherence += log((u_ij + 1) / u_i)
                # print(i, j, u_i, u_ij)
        print(f"coherence for cluster: {cluster}: {coherence}")
        coherence_sum += coherence
    coherence_sum /= len(top_item_infos)
    return coherence_sum


def get_all_items(
    raw_sequences: Union[Dict[str, List[str]], ChainMap[str, List[str]]]
) -> List[str]:
    st = set()
    for seq in raw_sequences.values():
        for e in seq:
            st.add(e)
    return list(st)
