from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

from model import Model, MyDoc2Vec


class Trainer:
    def __init__(
        self,
        dataset: Dataset,
        num_seq: int,
        num_item: int,
        model: str = 'model',
        d_model: int = 100,
        batch_size: int = 5,
        epochs: int = 100,
        lr: float = 0.00005,
    ):
        self.epochs = epochs

        if model == 'model':
            self.model = Model(num_seq, num_item, d_model)
        elif model == 'doc2vec':
            self.model = MyDoc2Vec(num_seq, num_item, d_model)
        else:
            assert False, f'{model} is not a model name.'

        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.dataset = dataset
        self.data_loader = DataLoader(dataset, batch_size=batch_size)

    def train(self):
        self.model.train()
        losses = []
        for epoch in range(self.epochs):
            total_loss = 0
            for data in self.data_loader:
                seq_index, item_indicies, target_index = data

                h = self.model.forward(seq_index, item_indicies)
                loss = F.cross_entropy(h, target_index)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            total_loss /= len(self.data_loader)
            if epoch % 10 == 0:
                print(epoch, total_loss)

            losses.append(total_loss)

        return losses

    @torch.no_grad()
    def test(self):
        self.model.eval()
        for data in self.data_loader:
            seq_index, item_indicies, target_index = data
            h = self.model.forward(seq_index, item_indicies)
            print(h[0:100:20], target_index[0:100:20])
            break
