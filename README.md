# 卒業研究

## Setup

### Requirements

```
$ python3 --version
3.10.4

$ poetry --version
1.2.1
```

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
- コメントを日本語にする
- `Trainer.attention_weights_to_...`のリファクタリング
- ログをわかりやすく
- ちゃんと`nn.Embedding`とアイテムの順番が対応しているか確かめる
- `dropout`の追加
- 顧客の属性情報の埋め込みの追加
