import os
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def show_top_items(
    top_item_infos: List[Tuple[List[str], List[str]]]
) -> None:
    num_cluster = len(top_item_infos)
    num_top_item = len(top_item_infos[0][0])
    axes = []
    fig = plt.figure(figsize=(8, 12))

    for c, (top_items, ratios) in enumerate(top_item_infos):
        # for i, item in enumerate(item_le.inverse_transform(top_items)):
        for i, (item, ratio) in enumerate(zip(top_items, ratios)):
            idx = c * num_top_item + i
            axes.append(fig.add_subplot(num_cluster, num_top_item, idx + 1))
            axes[-1].set_title(str(round(float(ratio), 3)))

            path = f"../data/hm/images/{item[:3]}/{item}.jpg"
            if os.path.exists(path):
                im = Image.open(path)
                im_list = np.asarray(im)
                plt.imshow(im_list)
            else:
                print(f'file not found at {path}')
            plt.axis('off')

    fig.tight_layout()
    plt.show()


top_items = []

with open('../data/cluster_list.txt', 'r') as f:
    infos: List[Tuple[List[str], List[str]]] = []
    for i in range(10):
        size = int(f.readline())
        ids, ratios = [], []
        for j in range(10):
            id, ratio = f.readline().split()
            ids.append(id)
            ratios.append(ratio)
        top_items.append((ids, ratios))

show_top_items(top_items)
