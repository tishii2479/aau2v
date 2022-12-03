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
    - 大きすぎるのかも
- `dropout`の追加
- `item_meta_weight/indicies`のリファクタリング

## ISSUE:

- `to_sequential_data`の並列化
- ログをわかりやすく
- `pathlib`を使う
- コメントを日本語にする
    - `Analyst`の各関数にドキュメントを書く
    - Tensorの引数にサイズを書く
- `analyst`のリファクタリング
    - `runner`に名前を変える
    - `onTrainStart`、`onEpochEnd`などのコールバックを受け入れる
- 不要なオプションを消す
    - `add_positional_encoding`、`use_learnable_embedding`など
- 必要なディレクトリを自動で作る
- モデルの引数にデフォルト値を足す