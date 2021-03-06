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
学習は以下の設定で行った。
|item|value|
|-|-|
|batch_size|32|
|epochs|100|
|lr|0.001|
|weight_decay|0.0005|
|gamma|0.95|

論文では最適化手法として SGD を用いていたが、Adam と比較したところ、学習の進度（loss の減少速度）は Adam のほうが早いようだったので、本試行では Adam を使用した。 

学習率をスケジューリングし、学習率 = lr * (gamma ^ epoch) になるように学習を行った。

学習は論文準拠となるように実施する予定だったが、loss 計算で pred bbox と gt bbox をマッチングさせる際、iou_thresh = 0.5 という値だと、学習の最初の段階で gt bbox とマッチングするものが見つからず、voidとして学習が進んでしまう結果となり、特に小さい物体の検出に全体的に失敗した。

そこで iou_thresh = 0.25 として学習を行った。

結果を見ると、以下の推論結果に示したように大〜中程度の大きさの物体は検出ができている。

![image.png](images/dog.png)
![image.png](images/train.png)
![image.png](images/cat.png)

ただし、non maximum suppression を適用したにも関わらず pred bbox が複数個生じており、localization loss の誤差は残った。

論文では計5万 epoch 学習を行ったと書いてあるので、そのくらい学習を行えば localization loss も低下し、今以上に良い精度で検出ができる可能性はある。

実際、以下の tensorboard での出力結果（loss のグラフ）を見ても、微小ではあるものの減少は続いている。

![image.png](images/loss_train.png)
![image.png](images/loss_validation.png)

しかし、予算の都合上、本試行はここで終了とする。

本試行での定量評価の結果は以下の通りである。bottle や potted plant などの小さい物体に関して特に精度が低かったものの、前回作成した yolo モデルよりは高い精度を出すことができた。

|label|average precision|
|-|-|
|aeroplane|0.457|
|bicycle|0.27|
|bird|0.33|
|boat|0.181|
|bottle|0.044|
|bus|0.453|
|car|0.279|
|cat|0.635|
|chair|0.046|
|cow|0.231|
|diningtable|0.251|
|dog|0.558|
|horse|0.565|
|motorbike|0.401|
|person|0.26|
|pottedplant|0.074|
|sheep|0.177|
|sofa|0.298|
|train|0.593|
|tvmonitor|0.177|
|**mean**|**0.314**|

論文で提示されているスコアとの開きが大きいが、各クラス間でのスコアの大小関係は似た傾向を示しているため、全体的に学習が不十分なだけである可能性が高い。

単に学習回数が足りないだけか、あるいは学習方法が不適切なのか現時点では断定できないが、とりあえずスクラッチで実装し形になったので、今回はこれで完成とする。

今後はより最新のモデルの実装を試しながら、本実装の問題点や課題を見出し、より良いモデルの構築を目指していきたい。

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
SSD モデルは [論文](https://arxiv.org/pdf/1512.02325.pdf) 準拠で作成した。<br>
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
デフォルトBBOXの大きさ、アスペクト種数、個数は検出器 (モデルアーキテクチャにおける `detectors` の各 module を指す)ごとに異なっている。<br>
本実装では、論文の記載をもとに以下のようにデフォルトBBOXを作成した。

| 検出器 | BBOXの大きさ | BBOXのアスペクト種数 | BBOXの個数 |
|-|-|-|-|
|det_4_3| 0.20 | 4 | 38 * 38 * 4 = 5776 |
|det_7_1| 0.34 | 6 | 19 * 19 * 6 = 2166 |
|det_8_2| 0.48 | 6 | 10 * 10 * 6 = 600 |
|det_9_2| 0.62 | 6 | 5 * 5 * 6 = 150 |
|det_10_2| 0.76 | 4 | 3 * 3 * 4 = 36 |
|det_11_2| 0.90 | 4 | 1 * 1 * 4 = 4 |

BBOXの個数の総和は 8732 であり、一画像に対して 8732 個の BBOX を予測している。<br>
詳細の実装は [ここ](src/model/ssd.py) の _get_default_bboxes 関数を参照。

<a id="anchor6"></a>

# 損失関数
論文では以下の損失関数で損失を計算している。<br>
<img src="https://latex.codecogs.com/gif.latex?L(x,&space;c,&space;l,&space;g)&space;=&space;\frac{1}{N}(L_{conf}(x,&space;c)&space;&plus;&space;\alpha&space;L_{loc}(x,&space;l,&space;g))">

各項：<br>
<img src="https://latex.codecogs.com/gif.latex?L_{loc}(x,&space;l,&space;g)&space;=&space;\sum^{N}_{i&space;\in&space;Pos}&space;\sum_{m&space;\in&space;\{cx,&space;cy,&space;w,&space;h&space;\}}&space;x^{k}_{ij}smooth_{L1}(l^{m}_{i}&space;-&space;\hat{g}^{m}_{j})">

<img src="https://latex.codecogs.com/gif.latex?L_{conf}(x,&space;c)&space;=&space;-\sum^{N}_{i&space;\in&space;Pos}&space;x^{p}_{ij}log(\hat{c}^{p}_{i})&space;-\sum_{i&space;\in&space;Neg}&space;log(\hat{c}^{0}_{i})">

本実装では、上記の損失計算を以下の手順で実施した。
1. **デフォルトBBOX (shape: (P, 4)) と正解BBOX (shape: (N, G, C)) のマッチング**
    * デフォルトBBOX、正解BBOXの jaccard係数 を計算。テンソルを拡張し総当り的な計算を行い、shape: (N, P, G) のテンソルを計算結果として得る。
    * 閾値より大きいかどうか（Positive or Negative）を判定。shape: (N, P, G) の1, 0からなるテンソルを得る。
1. **Localization loss の計算**
    * smooth L1 値を計算後、1 のテンソルを掛けることで Positive のみを残す。
1. **Confidence loss (Positive項) の計算**
    * softmax cross entropy 値を計算後、1 のテンソルを掛けることで Positive のみを残す。
1. **Confidence loss (Negative項) の計算**
    * 1 のテンソルから、Negative な BBOX かどうかを判定。shape: (N, P) の1, 0からなるテンソルを得る。
    * softmax cross entropy 値を計算後、このテンソルを掛けることで Negative のみを残す。
1. **Hard Negative Mining の実施**
    * Hard Negative Mining を実施し、2 3 4で残った値のうち、Loss計算に含めるものはどれかを判定。shape: (N, P) の1, 0からなるテンソルを2つ(Pos, Neg)得る。
1. **Loss 合計値の計算**
    * 2 3 4 で得られたテンソルに、5 のテンソルをかけ、Loss計算に含めるものだけを残す。
    * 残った値をバッチ単位で合算し、平均を取る。

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
