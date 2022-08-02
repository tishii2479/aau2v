from typing import List
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from torch import Tensor


def visualize_cluster(
    features: List[Tensor],
    num_cluster: int,
    cluster_labels: List[int],
    answer_labels: List[int]
):
    r'''
    Visualize cluster to 2d
    '''
    pca = PCA(n_components=2)
    pca.fit(features)
    pca_features = pca.fit_transform(features)

    colors = cm.rainbow(np.linspace(0, 1, num_cluster))

    plt.figure()

    for i in range(pca_features.shape[0]):
        plt.scatter(x=pca_features[i, 0], y=pca_features[i, 1],
                    color=colors[cluster_labels[i]], marker='$' + str(answer_labels[i]) + '$')

    plt.show()
