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
    - `runner`と`analyst`に分離する
    - `onTrainStart`、`onEpochEnd`などのコールバックを受け入れる
- 不要なオプションを消す
    - `add_positional_encoding`、`use_learnable_embedding`など
- 必要なディレクトリを自動で作る
- モデルの引数にデフォルト値を足す
- `eval.py`を作る
- `meta_indicies/weights`のリファクタリング　

## TODO:

- `dropout`の追加
- 補助情報にない分布を作って、それを抽出できるか
- 学習を別々にやる
    - アイテムが先に学習される
    - 系列は後で学習される
- 重みの正規化
- `scale_grad_by_freq`の調査
- 特徴が分離できているか確かめる
    - 固有の埋め込み表現からどの程度性別の影響が取り除かれているか
