import os
from math import log
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from torch import Tensor


def visualize_cluster(
    features: List[Tensor],
    num_cluster: int,
    cluster_labels: List[int],
    answer_labels: Optional[List[int]] = None
) -> None:
    r'''
    Visualize cluster to 2d
    '''
    pca = PCA(n_components=2)
    pca.fit(features)
    pca_features = pca.fit_transform(features)

    colors = cm.rainbow(np.linspace(0, 1, num_cluster))

    plt.figure()

    for i in range(pca_features.shape[0]):
        if answer_labels is not None:
            marker = '$' + str(answer_labels[i]) + '$'
        else:
            marker = '.'
        plt.scatter(x=pca_features[i, 0], y=pca_features[i, 1],
                    color=colors[cluster_labels[i]], marker=marker)
    plt.show()


def visualize_loss(
    losses: List[float]
) -> None:
    plt.plot(losses)
    plt.show()


def top_cluster_items(
    num_cluster: int,
    cluster_occurence_array: np.ndarray,
    cluster_size: List[int],
    num_top_item: int,
) -> List[Tuple[List[int], List[float]]]:
    r'''
    Args:
        num_cluster: number of clusters
        topic_indicies: list of sequence indicies for each topic
            shape: (num_topic, len(sequence_indicies))
        sequences: all sequence data
        num_top_item: number of item to list
        num_item: number of items in data
    Return:
        top items for each cluster
            shape: (num_topic, num_top_item, (item_list, item_ratio))
    '''
    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(cluster_occurence_array).toarray()

    top_items = []
    for cluster in range(num_cluster):
        # Get item index of top items which has largest item_count
        top_items_for_cluster = list(
            tf_idf[cluster].argsort()[::-1][:num_top_item])
        top_item_counts = np.sort(tf_idf[cluster])[::-1][:num_top_item]
        top_items_for_cluster_counts = \
            list(top_item_counts / cluster_size[cluster])
        top_items.append((top_items_for_cluster, top_items_for_cluster_counts))
    return top_items


def check_model_path(model_path: str) -> None:
    if os.path.exists(model_path):
        response = input(
            f'There is a file at {model_path}, but did not specify `load_model. Is it ok to' +
            'overwrite? [y/n] ')
        if response != 'y':
            exit(0)


def calc_cluster_occurence_array(
    num_cluster: int,
    cluster_labels: List[int],
    sequences: List[List[int]],
    num_item: int
) -> Tuple[np.ndarray, List[int]]:
    occurence_array = np.zeros((num_cluster, num_item))
    cluster_size = [0] * num_cluster
    for i, sequence in enumerate(sequences):
        cluster_size[cluster_labels[i]] += 1
        for item_index in set(sequence):
            occurence_array[cluster_labels[i]][item_index] += 1
    return occurence_array, cluster_size


def calc_sequence_occurence_array(
    sequences: List[List[int]],
    num_item: int
) -> np.ndarray:
    occurence_array = np.zeros((len(sequences), num_item))
    for i, sequence in enumerate(sequences):
        for item_index in set(sequence):
            occurence_array[i][item_index] += 1
    return occurence_array


def calc_coherence(
    sequence_occurence_array: np.ndarray,
    top_item_infos: List[Tuple[List[int], List[float]]]
) -> float:
    coherence_sum = 0.
    for cluster, (top_items, _) in enumerate(top_item_infos):
        coherence = 0.
        for i in top_items:
            for j in top_items:
                # ignore duplicate pairs
                if i <= j:
                    continue
                d_ij = (sequence_occurence_array[:, i] > 0) & (sequence_occurence_array[:, j] > 0)
                u_ij = d_ij.sum()
                d_i = sequence_occurence_array[:, i]
                u_i = d_i.sum()
                coherence += log((u_ij + 1) / u_i)
                # print(i, j, u_i, u_ij)
        print(f'coherence for cluster: {cluster}: {coherence}')
        coherence_sum += coherence
    coherence_sum /= len(top_item_infos)
    return coherence_sum
