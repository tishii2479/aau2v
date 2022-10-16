# 卒業研究

## Setup

### Requirements

- python3 = 3.10.4
- poetry = 1.2.1

### Setup environment

```shell
$ poetry shell
$ poetry install
```

### Train model

```shell
$ python3 train.py --epochs=5
```

## TODO:

- 重みの初期値を変える
- 位置エンコーディングを試す
- User2Vecを動かす
- コメントを日本語にする
- load_modelのログの追加
- `Trainer.attention_weights_to_...`のリファクタリング
