from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
from data import SequenceDataset

from model import Model, MyDoc2Vec


class Trainer:
    def __init__(
        self,
        dataset: SequenceDataset,
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
            self.model = Model(num_seq, num_item, d_model,
                               dataset.sequences)
        elif model == 'doc2vec':
            self.model = MyDoc2Vec(
                num_seq, num_item, d_model, dataset.sequences)
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

                loss = self.model.forward(
                    seq_index, item_indicies, target_index)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            total_loss /= len(self.data_loader)
            if epoch % 1 == 0:
                print(epoch, total_loss)

            losses.append(total_loss)

        return losses

    @torch.no_grad()
    def test(self):
        self.model.eval()
        for data in self.data_loader:
            seq_index, item_indicies, target_index = data
            loss = self.model.forward(seq_index, item_indicies, target_index)
            print(loss)
            break
