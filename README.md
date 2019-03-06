# Yolov3-Darknet-Training-Dataset

[Yolov3 Homepage 主页](https://pjreddie.com/darknet/yolo/)

Yolov3 可以处理物体分类 和 物体检测 2种任务

## 安装 Darknet

```
git clone https://github.com/pjreddie/darknet
cd darknet
make
```

## 物体检测 Object Detection

### 使用预训练的模型进行检测 (直接做推断)

安装Darknet之后, 在克隆好的文件目录中, 存在cfg/子目录. 可以将[预训练的权重文件](https://pjreddie.com/media/files/yolov3.weights)下载到该目录。


也可以直接运行如下命令进行下载:

```
wget https://pjreddie.com/media/files/yolov3.weights
```
