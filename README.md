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
- コメントを日本語にする
- `Trainer.attention_weights_to_...`のリファクタリング
- ログをわかりやすく
- ちゃんと`nn.Embedding`とアイテムの順番が対応しているか確かめる
- `dropout`の追加
- `to_sequential_data`の並列化
- 複数ジャンルの映画への対応
    - 複数ある場合には平均を取る
- `Q_k` を取り除く
