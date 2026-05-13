import os
import random
import xml.etree.ElementTree as ET
import numpy as np
from utils.utils import get_classes
import glob
"""
0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的2007_train.txt、2007_val.txt
1代表获得VOCdevkit/VOC2007/ImageSets里面的txt
2代表获得训练用的2007_train.txt、2007_val.txt

python coco_annotation_ExDark.py

"""
annotation_mode = 0
classes_path = 'model_data/classes_Exdark.txt'


VOCdevkit_path = 'Exdark_YOLO_training_82_split'
VOCdevkit_sets = [('Exdark', 'train'), ('Exdark', 'val')]

classes, _ = get_classes(classes_path)
photo_nums = np.zeros(len(VOCdevkit_sets))
nums = np.zeros(len(classes))


def convert_annotation(image_set, image_id, list_file):
    txtfile=glob.glob(os.path.join('Exdark_YOLO_training_82_split/labels/%s'%image_set, image_id+'*'))[0]
    
    with open(txtfile, 'r') as ftxt:
        lines = ftxt.readlines()
    for line in lines:
        line_l=line.strip('\n').split(' ')
        '''
                l - 顶部x坐标
                t - 顶部y坐标
                w - bounding box的宽
                h - bounding box的高
                Exdark [xmin,ymin,width,height] 需要转换为 [xmin,ymin,xmax,ymax]格式
        '''
        cls_id=line_l[0]
        line_l[1]=line_l[1]
        line_l[2]=line_l[2]
        line_l[3]=str(int(line_l[1])+int(line_l[3]))
        line_l[4]=str(int(line_l[2])+int(line_l[4]))
        
        list_file.write(" " + ",".join([str(a) for a in line_l[1:5]]) + ',' + str(cls_id))



if __name__ == "__main__":
    random.seed(0)
    train_img_path='Exdark_YOLO_training_82_split/images/train'
    val_img_path='Exdark_YOLO_training_82_split/images/val'
   
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")
    if annotation_mode == 0 or annotation_mode == 1:#保存训练集、验证集、测试集txt
        print("Generate txt in ImageSets.")
        image_name_train=os.listdir(train_img_path)
        image_name_val=os.listdir(val_img_path)
        saveBasePath    = 'Exdark_YOLO_training_82_split'

        num_train = len(image_name_train) 
        num_val=len(image_name_val) 
        
        ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
        fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
        for j in range(num_train):
            name=image_name_train[j].split('.')[0]+'\n'
            ftrain.write(name)
        for j in range(num_val):
            name=image_name_val[j].split('.')[0]+'\n'
            fval.write(name)
          
       
    
        ftrain.close()  
        fval.close()  

        print('训练集多少张：',num_train)# 5896
        print('验证集多少张：',num_val)# 1467
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate Exdark_train.txt and Exdark_val.txt  for train.")
        type_index = 0
        for year, image_set in VOCdevkit_sets:
            image_ids=[]
            
            if image_set=='train':
                image_name=os.listdir(train_img_path)
                for i in image_name:
                    image_ids.append(i.split('.')[0])
            elif image_set=='val':
                image_name=os.listdir(val_img_path)
                for i in image_name:
                    image_ids.append(i.split('.')[0])
            
            
            list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                img_file_path = glob.glob(os.path.join('Exdark_YOLO_training_82_split/images/%s'%image_set, image_id+'*'))[0]
                list_file.write(img_file_path)
                convert_annotation(image_set, image_id, list_file)
                list_file.write('\n')
              
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("Generate Exdark_train.txt and Exdark_val.txt for train done.")


        if photo_nums[0] <= 500:
            print("训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。")

        if np.sum(nums) == 0:
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")

