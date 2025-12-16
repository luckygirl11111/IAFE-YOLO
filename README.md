# Illumination-adaptive Feature Enhancement for Low-light Object Detection

## 训练步骤
### 训练ExDark数据集
1. 数据集的准备   
**本文使用COCO格式进行训练，训练前需要下载好ExDark的数据集，解压后放在根目录**

2. 数据集的处理   
(1) 修改coco_annotation.py里面的annotation_mode=2。
(2) 建立一个classes_Exdark.txt，里面写ExDark数据集所需要区分的类别。model_data/classes_Exdark.txt文件内容为：      
 ```python
 Bicycle
 Boat
 Bottle
 Bus
 Car
 Cat
 Chair
 Cup
 Dog
 Motorbike
 People
 Table
 ```
(3) 修改coco_annotation.py中的classes_path，使其对应classes_Exdark.txt，并运行coco_annotation.py，生成根目录下的Exdark_train.txt和Exdark_val.txt，包含训练/验证图片路径和标签。   

3. 开始网络训练  
**训练的参数较多，均在train_in_Exdark.py中**  
**修改classes_path用于指向检测类别所对应的txt，修改model_path用于加载YOLOv5-l的预训练权重，修改train_annotation_path  = 'labels/Exdark_train.txt'，修改val_annotation_path = 'labels/Exdark_val.txt'**  
修改完以上的路径后就可以运行train_in_Exdark.py开始训练了，在训练多个epoch后，权值会生成在log_YOLOV5l_in_Exdark文件夹中。

## 预测步骤
### 使用自己训练的权重
1. 在yolo_YOLOv5_in_Exdark.py文件里面，修改model_path，classes_path和dir_origin_path使其对应训练好的文件；**model_path对应log_YOLOV5l_in_Exdark文件夹下面的权值文件，classes_path是model_path对应分的类，dir_origin_path是ExDark数据集存放的路径**。
2. 运行predict_in_Exdark.py，即可获得评估结果(mAP50),以及检测的结果存在dir_save_path下

