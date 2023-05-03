# 卒業研究

## 環境構築

```
$ python3 --version
3.10.4

$ poetry --version
1.2.1
```

- パッケージマネージャに`poetry`（ https://python-poetry.org/docs/ ）を使用

### 必要ライブラリのインストール

```shell
$ poetry install
```

## サンプルコード

- [example.ipynb](/example.ipynb)

## モデルの学習

```shell
$ poetry run python3 src/train.py

or

$ poetry shell
$ python3 src/train.py
```

### 学習時の設定

```
$ poetry run python3 src/train.py --help

usage: train.py [-h] [--model-name {attentive,old-attentive,doc2vec}]
                [--dataset-name {toydata-paper,toydata-small,hm,movielens,movielens-simple,movielens-equal-gender,20newsgroup,20newsgroup-small}]
                [--d-model D_MODEL]
                [--max-embedding-norm MAX_EMBEDDING_NORM]
                [--init-embedding-std INIT_EMBEDDING_STD] [--window-size WINDOW_SIZE]
                [--negative_sample_size NEGATIVE_SAMPLE_SIZE] [--batch-size BATCH_SIZE]
                [--epochs EPOCHS] [--lr LR] [--verbose] [--load-model]
                [--ignore-saved-model] [--no-save-model] [--no-load-dataset]
                [--no-save-dataset] [--model-dir MODEL_DIR] [--dataset-dir DATASET_DIR]

options:
  -h, --help            show this help message and exit
  --model-name {attentive,old-attentive,doc2vec}
                        使用するモデル
  --dataset-name {toydata-paper,toydata-small,hm,movielens,movielens-simple,movielens-equal-gender,20newsgroup,20newsgroup-small}
                        使用するデータセット
  --d-model D_MODEL     埋め込み表現の次元数
  --max-embedding-norm MAX_EMBEDDING_NORM
                        埋め込み表現のノルムの最大値
  --init-embedding-std INIT_EMBEDDING_STD
                        埋め込み表現を初期化する時に用いる正規分布の標準偏差
  --window-size WINDOW_SIZE
                        学習する際に参照する過去の要素の個数
  --negative_sample_size NEGATIVE_SAMPLE_SIZE
                        ネガティブサンプリングのサンプル数
  --batch-size BATCH_SIZE
                        バッチサイズ
  --epochs EPOCHS       エポック数
  --lr LR               学習率
  --verbose             ログを詳細に出すかどうか
  --load-model          `model_dir`からモデルのパラメータを読み込むかどうか
  --ignore-saved-model  `model_dir`にあるモデルのパラメータを無視するかどうか
  --no-save-model       `model_dir`にモデルを保存するかどうか
  --no-load-dataset     `datset_dir`からデータセットを読み込むかどうか
  --no-save-dataset     `dataset_dir`にデータセットを保存するかどうか
  --model-dir MODEL_DIR
                        モデルを保存するディレクトリ
  --dataset-dir DATASET_DIR
                        データセットを保存するディレクトリ
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
    - `logger`の使用
- `pathlib`を使う
- `eval.py`を作る
- pytorchのGPUを試す
    - https://zenn.dev/hidetoshi/articles/20220731_pytorch-m1-macbook-gpu
- `trainer_config`形の整理
    - `save_model`っている?
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

