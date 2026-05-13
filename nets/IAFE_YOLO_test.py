import torch
import torch.nn as nn
from nets.PENet_1 import *
from nets.ConvNext import ConvNeXt_Small, ConvNeXt_Tiny
from nets.CSPdarknet import C3, Conv, CSPDarknet
from nets.Swin_transformer import Swin_transformer_Tiny
# from einops import rearrange
import numpy as np
import kornia



class Separable_Convolution(nn.Module):#深度可分离卷积DWconv
    def __init__(self,c_in,c_out):
        super(Separable_Convolution, self).__init__()
        self.Depthwise_Convolution = nn.Conv2d(in_channels=c_in,
                                    out_channels=c_in,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=c_in)
        self.Pointwise_Convolution = nn.Conv2d(in_channels=c_in,
                                    out_channels=c_out,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.Depthwise_Convolution(input)
        out = self.Pointwise_Convolution(out)
        return out
    
class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
    
class FeedForward(nn.Module):#FFN前馈神经网络
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out
    
class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
    
class FIB(nn.Module):#交互模块
    def __init__(self,in_channel,out_channel):
        super(FIB, self).__init__()
        self.linear=nn.Linear(in_channel,out_channel)
        self.dwconv=Separable_Convolution(in_channel,out_channel)
        self.silu=nn.SiLU()
        self.FFN=FeedForward(dim=in_channel)
        self.norm = nn.LayerNorm(in_channel)
    def forward(self,feat1,feat2,feat3):
        B, C, H, W = feat1.shape
        feat1_new = feat1.flatten(2).transpose(1, 2)
        feat2_new = feat2.flatten(2).transpose(1, 2)
        feat3_new=  feat3.flatten(2).transpose(1, 2)

        x=self.norm(feat1_new)
        feat1_1=self.linear(x)
        feat1_1 = feat1_1.view(B, H, W, C).permute(0, 3, 1, 2)
        feat1_1=self.dwconv(feat1_1)
  
        feat1_1=feat1_1.flatten(2).transpose(1, 2)
       
        feat1_2=self.silu(self.linear(x))

        feat2_new=self.linear(self.norm(feat2_new))
        feat2_new=feat2_new.view(B, H, W, C).permute(0, 3, 1, 2)
        feat2_new=self.dwconv(feat2_new)
        feat2_new=feat2_new.flatten(2).transpose(1, 2)

        feat3_new=self.linear(self.norm(feat3_new))
        feat3_new=feat3_new.view(B, H, W, C).permute(0, 3, 1, 2)
        feat3_new=self.dwconv(feat3_new)
        feat3_new=feat3_new.flatten(2).transpose(1, 2)

        out=feat1_1*feat1_2+feat1_2*feat2_new+feat1_2*feat3_new
        out=feat1+self.linear(out).view(B, H, W, C).permute(0, 3, 1, 2)
        
        return out


#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, backbone='cspdarknet', pretrained=False, input_shape=[640, 640]):
        super(YoloBody, self).__init__()
        depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        #-----------------------------------------------#
        self.backbone_name  = backbone
        self.penet=PENet()
        if backbone == "cspdarknet":
            #---------------------------------------------------#   
            #   生成CSPdarknet53的主干模型
            #   获得三个有效特征层，他们的shape分别是：
            #   80,80,256
            #   40,40,512
            #   20,20,1024
            #---------------------------------------------------#
            self.backbone   = CSPDarknet(base_channels, base_depth, phi, pretrained)
        else:
            #---------------------------------------------------#   
            #   如果输入不为cspdarknet，则调整通道数
            #   使其符合YoloV5的格式
            #---------------------------------------------------#
            self.backbone       = {
                'convnext_tiny'         : ConvNeXt_Tiny,
                'convnext_small'        : ConvNeXt_Small,
                'swin_transfomer_tiny'  : Swin_transformer_Tiny,
            }[backbone](pretrained=pretrained, input_shape=input_shape)
            in_channels         = {
                'convnext_tiny'         : [192, 384, 768],
                'convnext_small'        : [192, 384, 768],
                'swin_transfomer_tiny'  : [192, 384, 768],
            }[backbone]
            feat1_c, feat2_c, feat3_c = in_channels 
            self.conv_1x1_feat1 = Conv(feat1_c, base_channels * 4, 1, 1)
            self.conv_1x1_feat2 = Conv(feat2_c, base_channels * 8, 1, 1)
            self.conv_1x1_feat3 = Conv(feat3_c, base_channels * 16, 1, 1)
            
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3         = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1    = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2         = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2    = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1  = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2  = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        # 80, 80, 256 => 80, 80, 3 * (5 + num_classes) => 80, 80, 3 * (4 + 1 + num_classes)
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        # 40, 40, 512 => 40, 40, 3 * (5 + num_classes) => 40, 40, 3 * (4 + 1 + num_classes)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        # 20, 20, 1024 => 20, 20, 3 * (5 + num_classes) => 20, 20, 3 * (4 + 1 + num_classes)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)

  

        #135epo
        gamma = [0.8035668, 1.6044126] 
        sigma = [0.34896225, 0.24329378]
        
        bias=[0.0001,0.0001]
        self.sigma = torch.FloatTensor([sigma])
        self.gamma = torch.FloatTensor([gamma])
        self.bias = torch.FloatTensor(bias)
       
        self.norm=nn.BatchNorm2d(1,affine=True)#Ycbcr空间用这个
        self.conv1=nn.Conv2d(3, base_channels*4, 1, 1, bias=False)
        self.conv2=nn.Conv2d(3, base_channels*8, 1, 1, bias=False)
        self.conv3=nn.Conv2d(3, base_channels*8, 1, 1, bias=False)
        self.inter1=FIB(base_channels*4,base_channels*4)
        self.inter2=FIB(base_channels*8,base_channels*8)
        self.inter3=FIB(base_channels*8,base_channels*8)

    def ABTF(self,x):#可学习的自适应曝光
        gamma_a=self.gamma[0][0]
        gamma_c=self.gamma[0][1]
        
        g_a=torch.pow(x, gamma_a)
        mean_g_a=torch.pow(self.sigma[0][0], gamma_a)
        g_c=torch.pow(x, gamma_c)
        mean_g_c=torch.pow(self.sigma[0][1], gamma_c)

        y_a=g_a/((g_a+(1-g_a)*mean_g_a)+self.bias[0])
        y_c=g_c/((g_c+(1-g_c)*mean_g_c)+self.bias[1])

        y_a = y_a.type(torch.FloatTensor).to(x.device)
        y_a = self.norm(y_a)

        y_c = y_c.type(torch.FloatTensor).to(x.device)
        y_c = self.norm(y_c)
        return y_a,y_c#分别是过曝、欠曝
    
    def forward(self, x):
        _, _, H, W = x.shape

        #将image RGB转YCbCr
        y, cb, cr = torch.split(kornia.color.rgb_to_ycbcr(x), 1, dim=1)
       
		#RGB转YCbCr,只对Y分量进行调整曝光度
        over_expo_y,under_expo_y=self.ABTF(y)#分别是过曝、欠曝图像

        #将(Y,Cb,Cr)转回RGB空间
        over_expo = kornia.color.ycbcr_to_rgb(torch.cat([over_expo_y, cb, cr], dim=1)) 
        under_expo = kornia.color.ycbcr_to_rgb(torch.cat([under_expo_y, cb, cr], dim=1))
        

        over_enhance_img_ori=self.penet(over_expo)
        under_enhance_img_ori=self.penet(under_expo)

        pyr_1_over=self.conv1(over_enhance_img_ori[1][0])
        pyr_2_over=F.interpolate(over_enhance_img_ori[1][1], size=(40, 40), mode='nearest')
        pyr_2_over=self.conv2(pyr_2_over)
        pyr_3_over=F.interpolate(over_enhance_img_ori[1][2], size=(20, 20), mode='nearest')
        pyr_3_over=self.conv3(pyr_3_over)

        pyr_1_under=self.conv1(under_enhance_img_ori[1][0])
        pyr_2_under=F.interpolate(under_enhance_img_ori[1][1], size=(40, 40), mode='nearest')
        pyr_2_under=self.conv2(pyr_2_under)
        pyr_3_under=F.interpolate(under_enhance_img_ori[1][2], size=(20, 20), mode='nearest')
        pyr_3_under=self.conv3(pyr_3_under)


        over_enhance_img=over_enhance_img_ori[0]+over_expo
        under_enhance_img=under_enhance_img_ori[0]+under_expo
        
        

        over_fft = torch.fft.fft2(over_enhance_img, norm='backward')#傅里叶变换
        under_fft = torch.fft.fft2(under_enhance_img, norm='backward')#傅里叶变换
        mag_image = torch.abs(over_fft)#振幅变量
        pha_image = torch.angle(under_fft)#相位变量
        real_image_enhanced = mag_image * torch.cos(pha_image)#实部
        imag_image_enhanced = mag_image * torch.sin(pha_image)#虚部
        enhance_img = torch.fft.ifft2(torch.complex(real_image_enhanced, imag_image_enhanced), s=(H, W),
                                           norm='backward').real#逆傅里叶变换
        
        feat1, feat2, feat3 = self.backbone(enhance_img)
        if self.backbone_name != "cspdarknet":
            feat1 = self.conv_1x1_feat1(feat1)
            feat2 = self.conv_1x1_feat2(feat2)
            feat3 = self.conv_1x1_feat3(feat3)

        # 20, 20, 1024 -> 20, 20, 512
        P5          = self.conv_for_feat3(feat3)
        # 20, 20, 512 -> 40, 40, 512
        P5_upsample = self.upsample(P5)
        # 40, 40, 512 -> 40, 40, 1024
        feat2=self.inter2(feat2,pyr_2_over,pyr_2_under)
        P4          = torch.cat([P5_upsample, feat2], 1)
        # 40, 40, 1024 -> 40, 40, 512
        P4          = self.conv3_for_upsample1(P4)

        # 40, 40, 512 -> 40, 40, 256
        P4          = self.conv_for_feat2(P4)
        # 40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample(P4)
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        feat1=self.inter1(feat1,pyr_1_over,pyr_1_under)
        P3          = torch.cat([P4_upsample, feat1], 1)
        # 80, 80, 512 -> 80, 80, 256
        P3          = self.conv3_for_upsample2(P3)
        
        # 80, 80, 256 -> 40, 40, 256
        P3_downsample = self.down_sample1(P3)
        # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        P4 = torch.cat([P3_downsample, P4], 1)
        # 40, 40, 512 -> 40, 40, 512
        P4 = self.conv3_for_downsample1(P4)

        # 40, 40, 512 -> 20, 20, 512
        P4_downsample = self.down_sample2(P4)
        # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        P5=self.inter3(P5,pyr_3_over,pyr_3_under)
        P5 = torch.cat([P4_downsample, P5], 1)
        # 20, 20, 1024 -> 20, 20, 1024
        P5 = self.conv3_for_downsample2(P5)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,80,80)
        #---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,40,40)
        #---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,20,20)
        #---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)
       
    
        return out0, out1, out2

