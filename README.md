# Implementation of AAU2V (Attentive and Auxiliary-Informative User2Vec)

## Requirements

```shell
$ python3 --version
3.11.6

$ poetry --version
1.5.1
```

## Setup

### Install required libraries

```shell
$ poetry install
```

### Download movielens dataset

1. Download "MovieLens 1M Dataset" from https://grouplens.org/datasets/movielens/1m/.
2. Place `ml-1m` inside a new directory `./data`
3. Run `$ poetry run python3 src/preprocess_movielens.py`

## Experiments

To reproduce the experiment results, you can use jupyter notebooks inside directory `./notebooks`.

- `exp-4.ipynb` : Experiment against artifical data.
- `exp-5-1.ipynb` : Precision evaluation experiment against movielens dataset.
- `exp-5-2.ipynb` : User and auxiliary information analysis against movielens dataset.

## How to create your own dataset

You can create your own dataset by creating an instance of `aau2v.dataset.RawDataset`.
Refer `aau2v/dataset.py` for more detail.
Once you create your own `RawDataset`, you can follow the below procedure to train a model.

```python
window_size = 5
dataset = RawDataset(...) # Create your own RawDataset here.
dataset_center = SequenceDatasetCenter(dataset, window_size)

model = load_model(
    dataset_center=dataset_center,
    trainer_config=trainer_config,
    model_config=model_config,
)
trainer = PyTorchTrainer(
    model=model,
    dataset_center=dataset_center,
    trainer_config=trainer_config,
)
trainer.fit()
```

## Configs

```
$ poetry run python3 src/train.py --help
usage: train.py [-h] [--model-name {aau2v,user2vec}] [--dataset-name {toydata-paper,toydata-small,movielens}] [--d-model D_MODEL]
                [--max-embedding-norm MAX_EMBEDDING_NORM] [--init-embedding-std INIT_EMBEDDING_STD] [--window-size WINDOW_SIZE]
                [--negative_sample_size NEGATIVE_SAMPLE_SIZE] [--lr LR] [--no-weight-tying] [--no-meta] [--no-attention]
                [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--weight-decay WEIGHT_DECAY] [--verbose] [--no-save-model]
                [--model-dir MODEL_DIR] [--device DEVICE]

options:
  -h, --help            show this help message and exit
  --model-name {aau2v,user2vec}
                        The name of embedding model to train
  --dataset-name {toydata-paper,toydata-small,movielens}
                        The name of dataset to use for training
  --d-model D_MODEL     The dimension of embeddings
  --max-embedding-norm MAX_EMBEDDING_NORM
                        The maximum l2-norm of embedding representations
  --init-embedding-std INIT_EMBEDDING_STD
                        The standard deviation of the normal distribution used when initializing embeddings
  --window-size WINDOW_SIZE
                        The number of elements referenced during training (window_size)
  --negative_sample_size NEGATIVE_SAMPLE_SIZE
                        The sample size for negative sampling
  --lr LR               The learning rate when optimizing
  --no-weight-tying     Train model without weight-tying
  --no-meta             Train model without auxiliary information
  --no-attention        Train model without attention aggregation
  --batch-size BATCH_SIZE
                        The batch size
  --epochs EPOCHS       The epoch number for training
  --weight-decay WEIGHT_DECAY
                        The strenght of weight decay.
  --verbose             Whether to output logs in detail or not
  --no-save-model       Does not save model
  --model-dir MODEL_DIR
                        The directory to save model weights.
  --device DEVICE       The device on which the computation is performed
```
