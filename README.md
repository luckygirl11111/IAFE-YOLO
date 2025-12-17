# Illumination-adaptive Feature Enhancement for Low-light Object Detection

## Steps for training the ExDark dataset
### I. Dataset Download   
**This paper adopts the COCO format for training. Before training, you need to download the ExDark dataset and place it in the root directory after unpacking.**
Download the dataset from the [here](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset).
We have already split the ExDark dataset with train set (80%) and test set (20%), see paper [MAET (ICCV 2021)].

### II. Dataset Processing   
1. Modify the annotation_mode to 2 in the coco_annotation_ExDark.py file.
2. Create a classes_Exdark.txt file that lists the categories to be distinguished in the ExDark dataset. The content of the model_data/classes_Exdark.txt file is:      
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
3. Modify the classes_path in coco_annotation_ExDark.py to point to classes_Exdark.txt, then run coco_annotation_ExDark.py to generate Exdark_train.txt and Exdark_val.txt in the root directory. These files will contain the paths and labels of the training/validation images.   
```
$ python coco_annotation_ExDark.py
```

4. Training your own model  
There are numerous training parameters, all of which are set in train_in_Exdark.py. 
**Modify the classes_path to point to the TXT file corresponding to the detection categories.** 
**Modify the model_path to load the pre-trained weights for YOLOv5-l.** 
**Set train_annotation_path = 'labels/Exdark_train.txt' and val_annotation_path = 'labels/Exdark_val.txt'.** 
```
$ python train_in_Exdark.py
```

## Testing with pretrain model
1. In the yolo_YOLOv5_in_Exdark.py file, modify model_path, classes_path, and dir_origin_path to correspond to the trained files:
(1). model_path points to the weight file.
(2). classes_path corresponds to the classes for which model_path is trained.
(3). dir_origin_path is the path where the test set images of the ExDark dataset are stored.

3. Run predict_in_Exdark.py to obtain the evaluation results (mAP50), and the detection results will be saved in the dir_save_path directory.
```
$ python predict_in_Exdark.py
```

