# object-detection-torch

object detection by SSD with pytorch

# 目次

1. [概要](#anchor1)
1. [推論結果・評価結果](#anchor2)
1. [使用するデータセット](#anchor3)
1. [モデルアーキテクチャ](#anchor4)
1. [デフォルトBBOX作成手順](#anchor5)
1. [損失関数](#anchor6)
1. [スクリプト構成](#anchor7)
1. [処理実施手順](#anchor8)

<a id="anchor1"></a>

# 概要

物体検出モデル及び一連のスクリプトを `pytorch` をベースにして作成する。

<a id="anchor2"></a>

# 推論結果・評価結果

準備中

<a id="anchor3"></a>

# 使用するデータセット

PASCAL VOC (Visual Object Classes) 2007, 2012を使用。

[参考ブログ](https://www.sigfoss.com/developer_blog/detail?actual_object_id=247)

> Pascal VOC 2007および2012の16, 551枚のtrainval画像を用いて学習し、Pascal VOC 2007の4, 952枚のtest画像を用いて評価する手法を使っていくことにします。

この方法を参考にしてみる。

学習データ：

* data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt
* data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt

テストデータ：

* data/VOCdevkit/VOC2007/ImageSets/Main/test.txt

<a id="anchor4"></a>

# モデルアーキテクチャ

準備中

<a id="anchor5"></a>

# デフォルトBBOX作成手順

準備中

<a id="anchor6"></a>

# 損失関数

準備中

<a id="anchor7"></a>

# スクリプト構成

``` 
src
├dataset.py     ・・・元データの加工・データセットのクラス
├evaluate.py    ・・・評価用のスクリプト
├inference.py   ・・・推論用のスクリプト
├labelmap.json  ・・・検出対象の一覧
├model.py       ・・・モデルの定義
└train.py       ・・・学習用のスクリプト
```

<a id="anchor8"></a>

# 処理実施手順

## 1. 環境構築

本スクリプトはEC2(p2.xlarge)での実施を想定している。

インスタンス起動後、[ここ](https://github.com/rs1004/tips/blob/master/setup/set_gpu_and_docker.md)を参考に GPU 設定、Docker 設定を行う。

## 2. Docker ビルド・起動

以下でDockerをビルド・起動する。

``` 
docker build -t gpu_env --rm=true docker/
docker run --shm-size=20g --gpus all -it --rm -v /work/object-detection-torch/:/work --name od gpu_env
```

## 3. 学習実行

学習は `train.py` で行う。以下実行時のパラメータを記載。

``` 
usage: train.py [-h] [--imsize IMSIZE] [--grid_num GRID_NUM]
                [--bbox_num BBOX_NUM] [--class_num CLASS_NUM]
                [--l_coord L_COORD] [--l_noobj L_NOOBJ]
                [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                [--save_period SAVE_PERIOD] [--save_path SAVE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --imsize IMSIZE
  --grid_num GRID_NUM
  --bbox_num BBOX_NUM
  --class_num CLASS_NUM
  --l_coord L_COORD
  --l_noobj L_NOOBJ
  --batch_size BATCH_SIZE
  --epochs EPOCHS
  --save_period SAVE_PERIOD
  --save_path SAVE_PATH
```

`batch_size` , `epochs` 以外はデフォルト値を想定。
