# Yolov3-Darknet-Training-Dataset

[Yolov3 Homepage 主页](https://pjreddie.com/darknet/yolo/)

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

检测摄像头视频, [Darknet编译时需要启用CUDA和OpenCV]()