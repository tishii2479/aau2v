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
- コメントを日本語にする
    - `Analyst`の各関数にドキュメントを書く
    - Tensorの引数にサイズを書く
- ログをわかりやすく
- ちゃんと`nn.Embedding`とアイテムの順番が対応しているか確かめる
- `dropout`の追加
- `to_sequential_data`の並列化
- 評価を考慮してデータをつくる
- データ作成あたりのリファクタリング
    - データの種類を指定できるようにする
    - データをキャッシュする場所を変える
- `item_meta_weight/indicies`のリファクタリング
- `pathlib`を使う