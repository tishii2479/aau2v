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

- `to_sequential_data`の並列化
- ログをわかりやすく
    - エラーをちゃんと出す
- `pathlib`を使う
- コメントを日本語にする
    - `Analyst`の各関数にドキュメントを書く
    - Tensorの引数にサイズを書く
- `analyst`のリファクタリング
    - `similarity_*`の整理
    - `runner`と`analyst`に分離する
    - `onTrainStart`、`onEpochEnd`などのコールバックを受け入れる
- 不要なオプションを消す
    - `add_positional_encoding`、`use_learnable_embedding`など
    - `set_train_mode`も
- モデルの引数にデフォルト値を足す
- `eval.py`を作る
- `ArgumentParser`のenum系の引数に`choices`を追加する
- pytorchのGPUを試す
    - https://zenn.dev/hidetoshi/articles/20220731_pytorch-m1-macbook-gpu

## TODO:

- `dropout`の追加
- `scale_grad_by_freq`の調査
- 特徴が分離できているか確かめる
    - 固有の埋め込み表現からどの程度性別の影響が取り除かれているか
    - 性別を反転させたときにどうなるか
    - 男性の中で、好きな男性向けジャンルが異なるグループを作って、それをクラスタとして分けられるか
- 各要素の内積の大きさの比較
- `NormalizedEmbeddingLayer`に補助情報のグループのインデックスを与えて、グループごとに正規化する
    - その時は正規化はこまめにやる必要がありそう
- `init_embedding_dim`の値を0.3にする