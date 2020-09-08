<script async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>
<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [["\\(","\\)"] ],
    displayMath: [ ['$$','$$'], ["\\[","\\]"] ]
  }
 });
</script>

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
SSD モデルは [論文](https://arxiv.org/pdf/1512.02325.pdf) 準拠で作成した。</br>
本実装は `features` と `detectors` の2つのモジュールで構成した。
* `features`
  * 特徴量抽出層の群。ベースは vgg16_bn (act_5_3まで) を使用した。
* `detectors`
  * 検出器の群。`features` の所定の activation 直後のテンソルに対してかける。

<details><summary>アーキテクチャ</summary><div>

```
SSD(
  (features): ModuleDict(
    (conv_1_1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn_1_1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_1_1): ReLU(inplace=True)
    (conv_1_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn_1_2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_1_2): ReLU(inplace=True)
    (pool_1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv_2_1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn_2_1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_2_1): ReLU(inplace=True)
    (conv_2_2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn_2_2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_2_2): ReLU(inplace=True)
    (pool_2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv_3_1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn_3_1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_3_1): ReLU(inplace=True)
    (conv_3_2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn_3_2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_3_2): ReLU(inplace=True)
    (conv_3_3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn_3_3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_3_3): ReLU(inplace=True)
    (pool_3): MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False)
    (conv_4_1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn_4_1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_4_1): ReLU(inplace=True)
    (conv_4_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn_4_2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_4_2): ReLU(inplace=True)
    (conv_4_3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn_4_3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_4_3): ReLU(inplace=True)
    (pool_4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv_5_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn_5_1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_5_1): ReLU(inplace=True)
    (conv_5_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn_5_2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_5_2): ReLU(inplace=True)
    (conv_5_3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn_5_3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_5_3): ReLU(inplace=True)
    (conv_6_1): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn_6_1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_6_1): ReLU(inplace=True)
    (conv_7_1): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
    (bn_7_1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_7_1): ReLU(inplace=True)
    (conv_8_1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
    (bn_8_1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_8_1): ReLU(inplace=True)
    (conv_8_2): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (bn_8_2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_8_2): ReLU(inplace=True)
    (conv_9_1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
    (bn_9_1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_9_1): ReLU(inplace=True)
    (conv_9_2): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (bn_9_2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_9_2): ReLU(inplace=True)
    (conv_10_1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (bn_10_1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_10_1): ReLU(inplace=True)
    (conv_10_2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
    (bn_10_2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_10_2): ReLU(inplace=True)
    (conv_11_1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (bn_11_1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_11_1): ReLU(inplace=True)
    (conv_11_2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
    (bn_11_2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act_11_2): ReLU(inplace=True)
  )
  (detectors): ModuleDict(
    (det_10_2): Conv2d(256, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (det_11_2): Conv2d(256, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (det_4_3): Conv2d(512, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (det_7_1): Conv2d(1024, 150, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (det_8_2): Conv2d(512, 150, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (det_9_2): Conv2d(256, 150, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)
```

</div></details>

<a id="anchor5"></a>

# デフォルトBBOX作成手順
デフォルトBBOXの大きさ、アスペクト種数、個数は検出器 (モデルアーキテクチャにおける `detectors` の各 module を指す)ごとに異なっている。</br>
本実装では、論文の記載をもとに以下のようにデフォルトBBOXを作成した。

| 検出器 | BBOXの大きさ | BBOXのアスペクト種数 | BBOXの個数 |
|-|-|-|-|
|det_4_3| 0.20 | 4 | 38 * 38 * 4 = 5776 |
|det_7_1| 0.34 | 6 | 19 * 19 * 6 = 2166 |
|det_8_2| 0.48 | 6 | 10 * 10 * 6 = 600 |
|det_9_2| 0.62 | 6 | 5 * 5 * 6 = 150 |
|det_10_2| 0.76 | 4 | 3 * 3 * 4 = 36 |
|det_11_2| 0.90 | 4 | 1 * 1 * 4 = 4 |

BBOXの個数の総和は 8732 であり、一画像に対して 8732 個の BBOX を予測している。</br>
詳細の実装は [ここ](src/model/ssd.py) の _get_default_bboxes 関数を参照。

<a id="anchor6"></a>

# 損失関数
論文では以下の損失関数で損失を計算している。
* $ L(x, c, l, g) = \frac{1}{N}(L_{conf}(x, c) + αL_{loc}(x, l, g)) $
  * $ L_{loc}(x, l, g) = \sum^{N}_{i \in Pos} \sum_{m \in \{cx, cy, w, h \}} x^{k}_{ij}smooth_{L1}(l^{m}_{i} - \hat{g}^{m}_{j}) $
  * $ L_{conf}(x, c) = -\sum^{N}_{i \in Pos} x^{p}_{ij}log(\hat{c}^{p}_{i}) -\sum_{i \in Neg} log(\hat{c}^{0}_{i})  $

本実装では、上記の損失計算を以下の手順で実施した。
1. **デフォルトBBOX (shape: (P, 4)) と正解BBOX (shape: (N, G, C)) のマッチング**
  a. デフォルトBBOX（以下 df）、正解BBOX（以下gt）の jaccard係数 を計算。テンソルを拡張し総当り的な計算を行い、shape: (N, P, G) のテンソルを計算結果として得る。
  b. 閾値より大きいかどうか（Positive or Negative）を判定。shape: (N, P, G) の1, 0からなるテンソルを得る。
1. **Localization loss の計算**
  a. smooth L1 値を計算後、1. のテンソルを掛けることで Positive のみを残す。
1. **Confidence loss (Positive項) の計算**
  a. softmax cross entropy 値を計算後、1. のテンソルを掛けることで Positive のみを残す。
1. **Confidence loss (Negative項) の計算**
  a. 1. のテンソルから、Negative な BBOX かどうかを判定。shape: (N, P) の1, 0からなるテンソルを得る。
  b. softmax cross entropy 値を計算後、a. のテンソルを掛けることで Negative のみを残す。
1. **Hard Negative Mining の実施**
  a. Hard Negative Mining を実施し、2. 3. 4で残った値のうち、Loss計算に含めるものはどれかを判定。shape: (N, P) の1, 0からなるテンソルを2つ(Pos, Neg)得る。
1. **Loss 合計値の計算**
  a. 2. 3. 4. で得られたテンソルに、5. のテンソルをかけ、Loss計算に含めるものだけを残す。
  b. 残った値をバッチ単位で合算し、平均を取る。

詳細の実装は [ここ](src/model/ssd.py) の loss 関数を参照。

<a id="anchor7"></a>
# スクリプト構成

``` 
src
├augmentation   ・・・データ拡張の関数群
├model          ・・・モデルの定義（VGG16, SSD)
├dataset.py     ・・・元データの加工・データセットのクラス
├evaluate.py    ・・・評価用のスクリプト
├inference.py   ・・・推論用のスクリプト
├labelmap.json  ・・・検出対象の一覧
├train.py       ・・・学習用のスクリプト
└utils.py       ・・・共通関数のまとめ
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
usage: train.py [-h] [--purpose PURPOSE] [--imsize IMSIZE]
                [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--lr LR]
                [--weight_decay WEIGHT_DECAY] [--gamma GAMMA]
                [--num_workers NUM_WORKERS] [--result_dir RESULT_DIR]
                [--weights WEIGHTS] [--params PARAMS]

optional arguments:
  -h, --help            show this help message and exit
  --purpose PURPOSE
  --imsize IMSIZE
  --batch_size BATCH_SIZE
  --epochs EPOCHS
  --lr LR
  --weight_decay WEIGHT_DECAY
  --gamma GAMMA
  --num_workers NUM_WORKERS
  --result_dir RESULT_DIR
  --weights WEIGHTS
  --params PARAMS
```

`batch_size` , `epochs` 以外はデフォルト値を想定。

## 4. 推論実行

推論は `inference.py` で行う。以下実行時のパラメータを記載。

```
usage: inference.py [-h] [--imsize IMSIZE] [--batch_size BATCH_SIZE]
                    [--num_workers NUM_WORKERS] [--result_dir RESULT_DIR]
                    [--weights WEIGHTS]

optional arguments:
  -h, --help            show this help message and exit
  --imsize IMSIZE
  --batch_size BATCH_SIZE
  --num_workers NUM_WORKERS
  --result_dir RESULT_DIR
  --weights WEIGHTS
```

`batch_size` 以外はデフォルト値を想定。

## 5. 評価実行

評価は `evaluate.py` で行う。以下実行時のパラメータを記載。

```
usage: evaluate.py [-h] [--imsize IMSIZE] [--batch_size BATCH_SIZE]
                   [--num_workers NUM_WORKERS] [--result_dir RESULT_DIR]
                   [--weights WEIGHTS]

optional arguments:
  -h, --help            show this help message and exit
  --imsize IMSIZE
  --batch_size BATCH_SIZE
  --num_workers NUM_WORKERS
  --result_dir RESULT_DIR
  --weights WEIGHTS
```

`batch_size` 以外はデフォルト値を想定。
