# Yolov3-Training-Dataset

[Yolov3 Homepage 主页](https://pjreddie.com/darknet/yolo/)

[Pascal VOC Dataset Mirror](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)

[get_COCO_dataset](https://github.com/pjreddie/darknet/blob/master/scripts/get_coco_dataset.sh)


Yolov3 可以处理物体分类 和 物体检测 2种任务

## 0. 安装 Darknet

```
git clone https://github.com/pjreddie/darknet
cd darknet
make
```

## 1. 物体检测 Object Detection

### 1.A 推断

#### 1.A.1 使用预训练的模型进行检测 (直接做推断)

安装Darknet之后, 在克隆好的文件目录中, 存在cfg/子目录. 可以将[预训练的yolov3权重文件](https://pjreddie.com/media/files/yolov3.weights) 或者[yolov3-tiny的权重文件](https://pjreddie.com/media/files/yolov3-tiny.weights)下载到该目录。


也可以直接运行如下命令进行下载:

```
wget https://pjreddie.com/media/files/yolov3.weights

或者

wget https://pjreddie.com/media/files/yolov3-tiny.weights
```

#### 1.A.2 检测指定的图片

直接运行如下命令:

```
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg

或者

./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg
```

运行后, 显示(Yolov3-tiny)网络结构 :

<img src="./assets/tiny_detector_output.jpg" width=600>

(Yolov3-tiny)推断的结果 :

<img src="./assets/tiny_detector_result.jpg" width=400>

Darknet 打印出了运行时间, 检测到的物体, 和概率. 

Darknet并没有和OpenCV一起编译, 所以无法直接显示图片, 而是将图片保存在磁盘.

detect 命令是简写, 完整命令如下所示 :

```
./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg
```

通常Yolo 只显示物体概率值大于25%的物体, 如果需要显示更多物体, 可以将参数 -thresh <value>传给命令行：

```
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg -thresh 0
```

#### 1.A.3 摄像头实时监测

检测摄像头视频, [Darknet编译时需要启用CUDA和OpenCV](https://pjreddie.com/darknet/install/#cuda).

编译后, 运行 :

```
./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights
```

### 1.B 数据集上的训练

#### 1.B.1 获取数据, 可以下载[VOC 2007-2012的所有数据](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)

下载并解压数据:

```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```

#### 1.B.2 生成数据标注

Darknet读取每张图片的.txt标注文件, .txt文件中的数据格式如下:

```
<object-class> <x> <y> <width> <height>
```

x, y, width, height 是图像标注的坐标尺寸.

```
Darknet/scripts/voc_label.py   文件可以生成这样的.txt文件.
```

```
ls
2007_test.txt   VOCdevkit
2007_train.txt  voc_label.py
2007_val.txt    VOCtest_06-Nov-2007.tar
2012_train.txt  VOCtrainval_06-Nov-2007.tar
2012_val.txt    VOCtrainval_11-May-2012.tar
```

Darknet只能读取一个.txt文件, 里面包含了所有要训练的图片这样的标注信息.
```
cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
```

以上就是准备的要训练的数据.

#### 1.B.3 修改训练配置文件

修改 Darknet/cfg/voc.data 配置文件, 匹配刚才准备好的数据:

```
  1 classes= 20
  2 train  = <path-to-voc>/train.txt
  3 valid  = <path-to-voc>2007_test.txt
  4 names = data/voc.names
  5 backup = backup
```

#### 1.B.4 下载预训练的卷积权重文件

Yolo的训练过程是要先在小图片上通过分类任务训练卷积网络的特征提取权重, 

再用大图片对分类模型进行微调,

然后使用大图片训练检测任务.

可以使用Imagenet上预训练的卷积权重.

比如, 使用 [darknet53](https://pjreddie.com/darknet/imagenet/#darknet53)模型权重.

只需要下载 [卷积层权重](https://pjreddie.com/media/files/darknet53.conv.74).

并没有找到Yolov3-tiny的卷积层权重, 需要从头训练 yolov3-tiny. 🐍🐍🐍🐍


#### 1.B.5 训练新模型

```
./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74
```


================================

Yolov3-PyTorch

[YOLOv3: Training and inference in PyTorch](https://github.com/ultralytics/yolov3)

[Minimal PyTorch implementation of YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3.git) by eriklindernoren

