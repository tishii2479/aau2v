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
$ inv train-ml
```

## TODO:

- 重みの初期値を変える
- コメントを日本語にする
    - `Analyst`の各関数にドキュメントを書く
- `Trainer.attention_weights_to_...`のリファクタリング
- ログをわかりやすく
- ちゃんと`nn.Embedding`とアイテムの順番が対応しているか確かめる
- `dropout`の追加
- `to_sequential_data`の並列化
- データの指定を簡単にする
- cos類似度による比較
- 評価を考慮してデータをつくる