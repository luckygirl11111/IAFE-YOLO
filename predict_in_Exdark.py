import time
import torch
import cv2
import numpy as np
from PIL import Image
from utils.callbacks_mAP95 import  EvalCallback
from yolo_Our_in_Exdark_82split import YOLO, YOLO_ONNX
from nets.IAFE_YOLO_test import YoloBody
import os 
from utils.utils import (get_anchors, get_classes)
os.environ['CUDA_VISIBLE_DEVICES']='2'

'''

nohup python predict_in_Exdark.py > predict_Our_in_Exdark_82split_135epo.out&

'''
if __name__ == "__main__":
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #   'predict_onnx'      表示利用导出的onnx模型进行预测，相关参数的修改在yolo.py_423行左右处的YOLO_ONNX
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "Exdark_YOLO_training_82_split/images/val/"
    dir_enhance_path='end_to_end_enhanceimg/Exdark/Our_in_Exdark_enhanceimg_135epo'
    dir_save_path   = "test_result/test_result_Our_in_Exdark_82split_135epo/"
    
    #计算召回率和map
    test_annotation_path     = 'labels/Exdark_val82.txt'
    with open(test_annotation_path) as f:
        test_lines   = f.readlines()
    log_dir='log_YOLOV5l_in_Exdark/loss_test_Our_135epo_mAP95'
    if not os.path.isdir(log_dir):#train
        os.makedirs(log_dir)
    #---------------------------------------------------#
    #   获得种类和先验框的数量
    #---------------------------------------------------#
    
    model_path='saved_model/Exdark/IAFE-YOLO+_Exdark.pth'
    
    classes_path= 'model_data/classes_Exdark.txt'
    anchors_path='model_data/yolo_anchors.txt'
    anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape=[640, 640]
    class_names, num_classes  = get_classes(classes_path)
    anchors, num_anchors      = get_anchors(anchors_path)
    net    = YoloBody(anchors_mask, num_classes,'l')
    device      = torch.device('cpu')
    net.load_state_dict(torch.load(model_path, map_location=device),strict=False)
    net    = net.eval().cuda()
    path_new=['log_YOLOV5l_in_Exdark/loss_test_Our_135epo_mAP95/mAP50',
                      'log_YOLOV5l_in_Exdark/loss_test_Our_135epo_mAP95/mAP55',
                      'log_YOLOV5l_in_Exdark/loss_test_Our_135epo_mAP95/mAP60',
                      'log_YOLOV5l_in_Exdark/loss_test_Our_135epo_mAP95/mAP65',
                      'log_YOLOV5l_in_Exdark/loss_test_Our_135epo_mAP95/mAP70',
                      'log_YOLOV5l_in_Exdark/loss_test_Our_135epo_mAP95/mAP75',
                      'log_YOLOV5l_in_Exdark/loss_test_Our_135epo_mAP95/mAP80',
                      'log_YOLOV5l_in_Exdark/loss_test_Our_135epo_mAP95/mAP85',
                      'log_YOLOV5l_in_Exdark/loss_test_Our_135epo_mAP95/mAP90',
                      'log_YOLOV5l_in_Exdark/loss_test_Our_135epo_mAP95/mAP95']
    eval_callback   = EvalCallback(net, input_shape, anchors, anchors_mask, class_names, num_classes, test_lines, log_dir, True, \
                                            eval_flag=True, nms_iou=np.arange(0.5, 1.0, 0.05),MINOVERLAP=np.arange(0.5, 1.0, 0.05),period=1,map_out_path=path_new)
    
    #-------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #   
    #   heatmap_save_path仅在mode='heatmap'有效
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

 
    if mode != "predict_onnx":
        yolo = YOLO()
    else:
        yolo = YOLO_ONNX()
    
    if mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.JPEG', '.JPG', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)#/home/lujiajia/paper_code_1/object_detection_dataset/Exdark/ExDARk_YOLO3_training/images/test/2015_06649.JPEG
                name=img_name.split('.')[0]
                enhance_path=os.path.join(dir_enhance_path, name)+'.png'
                print('--------------------------------图片路径------------------------------')
                print(image_path)
                image       = Image.open(image_path)
                enhance=  Image.open(enhance_path)

                r_image     = yolo.detect_image(image,enhance)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
             
        eval_callback.on_epoch_end( 1, net) 
    
