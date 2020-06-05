# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from net.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from net.backbone import build_backbone
#from net.ASPP import ASPP
from net.DenseASPP import DenseASPP
from PIL import Image

class deeplabv3plus(nn.Module):
	def __init__(self, cfg):
		super(deeplabv3plus, self).__init__()
		self.backbone = None		
		self.backbone_layers = None
		#input_channel = 2368
		#self.aspp = ASPP(dim_in=input_channel, 
		#		dim_out=cfg.MODEL_ASPP_OUTDIM, 
		#		rate=16//cfg.MODEL_OUTPUT_STRIDE,
		#		bn_mom = cfg.TRAIN_BN_MOM)
		self.aspp = DenseASPP()
		self.dropout1 = nn.Dropout(0.5)
		#self.dropout1 = nn.Dropout(0.4)
		######修改dropout#######
		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE//4)

		indim = 256
		self.shortcut_conv = nn.Sequential(
				nn.Conv2d(128, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1, padding=cfg.MODEL_SHORTCUT_KERNEL//2,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),		
		)
		self.shortcut_conv1 = nn.Sequential(
			nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1,
					  padding=cfg.MODEL_SHORTCUT_KERNEL // 2, bias=True),
			SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),
		)
		self.cat_conv = nn.Sequential(
				#nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
                nn.Conv2d(2*cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
            ######################
				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
		############修改dropout#######
				nn.Dropout(0.5),
				#nn.Dropout(0.4),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.3),
		)
		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		self.end_conv = nn.Conv2d(cfg.MODEL_NUM_CLASSES, 1, 1, 1, padding=0)


		self.conv3 = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=False)
		self.conv4 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)
		self.conv5 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
		self.conv6 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn4 = SynchronizedBatchNorm2d(48)
		self.bn5 = SynchronizedBatchNorm2d(48)
		self.bn6 = SynchronizedBatchNorm2d(1)
		self.relu1=nn.ReLU(inplace=True)
		self.conv0 = nn.Conv2d(3, 3, kernel_size=7, stride=4, padding=3, bias=False)
		self.conv00 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)




		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, SynchronizedBatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
		self.backbone_layers = self.backbone.get_layers()
		 #自己添加#
		self.upsample_2=nn.UpsamplingBilinear2d(size=(57,76),scale_factor=None)
       
	def forward(self, x):

		#kipe = x
		#kipe = self.conv4(kipe)
		#kipe = self.bn4(kipe)
		#kipe = self.relu1(kipe)
		#kipe = self.conv5(kipe)
		#kipe = self.bn5(kipe)
		#kipe = self.relu1(kipe)
		#kipe = self.conv6(kipe)





		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()

#layer[-1]    torch.Size([4, 2048, 32, 32])
		#feature_aspp = self.aspp(layers[-1])
		#feature_aspp = self.dropout1(feature_aspp)
		#feature_aspp = self.upsample_sub(feature_aspp)
		feature_aspp = self.aspp(layers[-1])  #torch.Size([4, 2368, 32, 32])
		feature_aspp = self.dropout1(feature_aspp)
		feature_aspp = self.upsample_sub(feature_aspp)

		feature_aspp=self.shortcut_conv(feature_aspp)  #torch.Size([4, 48, 128, 128])
		feature_aspp = self.bn4(feature_aspp)
		feature_shallow = self.shortcut_conv1(layers[0])   #torch.Size([4, 48, 128, 128])
		feature_shallow = self.bn4(feature_shallow)
		#print(feature_aspp.size())
		#print(feature_shallow.size())
		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
		result = self.cat_conv(feature_cat)

		result = self.cls_conv(result)
		#result = self.end_conv(result)
		#result = torch.cat([kipe,result],1)
		result = self.bn6(result)
		result = self.relu1(result)
		#result = self.relu1(result)
		#result = self.conv3(result)

		result = self.upsample4(result)
		return result

