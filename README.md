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

## ISSUE:

- コメントを日本語にする
    - `Analyst`の各関数にドキュメントを書く
    - Tensorの引数にサイズを書く
- `analyst`のリファクタリング
    - `similarity_*`の整理
- `to_sequential_data`の並列化
- ログをわかりやすく
    - エラーをちゃんと出す
- `pathlib`を使う
- `eval.py`を作る
- pytorchのGPUを試す
    - https://zenn.dev/hidetoshi/articles/20220731_pytorch-m1-macbook-gpu
- `trainer_config`形の整理
    - `save_model`っている?
- `load_model`を`trainer`の外に出す
- `notebook`の整理

## TODO:

- `nn.Embedding.scale_grad_by_freq`の調査
- 特徴が分離できているか確かめる
    - 固有の埋め込み表現からどの程度性別の影響が取り除かれているか
    - 性別を反転させたときにどうなるか
    - 男性の中で、好きな男性向けジャンルが異なるグループを作って、それをクラスタとして分けられるか
- `NormalizedEmbeddingLayer`に補助情報のグループのインデックスを与えて、グループごとに正規化する
    - その時は正規化はこまめにやる必要がありそう
- 学習データの数と埋め込み表現のノルムに相関があるか確認する

