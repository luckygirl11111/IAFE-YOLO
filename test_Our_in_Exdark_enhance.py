#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import torch
import cv2
import numpy as np
from PIL import Image

from nets.IAFE_YOLO_test_enhanceimg import YoloBody
import os 
from utils.utils import (get_anchors, get_classes)
from utils.utils import cvtColor, preprocess_input
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

'''
python test_Our_in_Exdark_enhance.py 

'''

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    '''
        #   letterbox_image:该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，在多次测试后，发现关闭letterbox_image直接resize的效果更好
    '''

    w, h    = size
    iw, ih  = image.size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image,nw,nh

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
    dir_save_path   = "end_to_end_enhanceimg/Exdark/Our_in_Exdark_enhanceimg_135epo/"
    
    #计算召回率和map
    test_annotation_path     = 'labels/Exdark_val82.txt'
    with open(test_annotation_path) as f:
        test_lines   = f.readlines()
    
    if not os.path.isdir(dir_save_path):
        os.makedirs(dir_save_path)
    #---------------------------------------------------#
    #   获得种类和先验框的数量
    #---------------------------------------------------#
    model_path='saved_model/Exdark/ep135-loss0.055-val_loss0.057.pth'
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
    
   

    import os

    from tqdm import tqdm
    size = [640, 640]
    img_names = os.listdir(dir_origin_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.JPEG', '.JPG', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path  = os.path.join(dir_origin_path, img_name)
            print('--------------------------------图片路径------------------------------')
            print(image_path)
            image_ori       = Image.open(image_path)
            H,W=image_ori.size#640 393
         

            image, nw,nh = resize_image(image_ori, size) 
            
            image   = cvtColor(image)

            data_lowlight =np.expand_dims(np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1)), 0)
            data_lowlight = torch.from_numpy(data_lowlight)
           
            data_lowlight = data_lowlight.cuda()
            enhance_img = net(data_lowlight)
          
            
            

            enhance_img=enhance_img.permute(0, 2, 3, 1).cpu().data.numpy()#(1, 640, 640, 3)
            enhance_img=enhance_img[0,:,:,:]

            
            #   将灰条部分截取掉
            enhance_img = enhance_img[int((size[0] - nh) // 2): int((size[0] - nh) // 2 + nh), \
                 int((size[1] - nw) // 2): int((size[1] - nw) // 2 + nw)]
            
            # ---------------------------------------------------#
            #   进行图片的resize
            # ---------------------------------------------------#
            enhance_img = cv2.resize(enhance_img, (H, W), interpolation=cv2.INTER_LINEAR)
          

  
            img_name_enhance=img_name.split('.')[0]+'.png'
            img_name_enhance=os.path.join(dir_save_path, img_name_enhance)
            
            cv2.imwrite(img_name_enhance, enhance_img[:, :, [2,1,0]]*255)
          
            