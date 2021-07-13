import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

from functools import partial
from third_modules.batch_dynamic_conv import *
### Implementation of EAB block 
class STCeptionLayer_max(nn.Module):
	expansion = 4

	def __init__(self, planes,ratio = 4,dyn_network_ratio = 16, is_res = True,matrix_res = False):
		self.used_fn = planes
		self.ratio = ratio
		self.is_res = is_res
		self.matrix_res = matrix_res
		super(STCeptionLayer_max, self).__init__()
		self.used_fn = planes//ratio
		if self.ratio>1:
			self.compress_c1 = nn.Conv3d(planes, self.used_fn, kernel_size=(1,1,1), stride=(1,1,1),
								padding=(0,0,0), bias=False,groups=4)
			self.expand_c1 = nn.Conv3d( self.used_fn,planes, kernel_size=(1,1,1), stride=(1,1,1),
								padding=(0,0,0), bias=False,groups=4)
			self.expand_c1.weight.data = self.expand_c1.weight.data*0.01
		


		self.t_part_num = 3

		self.t_part2_conv = nn.Conv3d(self.used_fn// self.t_part_num, self.used_fn// self.t_part_num, kernel_size=(3,1,1), stride=(1,1,1),
							   padding=(1,0,0), bias=False)
		last_nft = self.used_fn  - self.used_fn// self.t_part_num*2
		
		self.t_part3_conv = nn.Conv3d(last_nft, last_nft, kernel_size=(3,1,1), stride=(1,1,1),
							   dilation=(2,1,1),padding=(2,0,0), bias=False)
	

		self.s_part_num = 3

		self.s_part2_conv = nn.Conv3d(self.used_fn//self.s_part_num , self.used_fn//self.s_part_num , kernel_size=(1,3,3), stride=(1,1,1),dilation=(1,1,1),
							   padding=(0,1,1), bias=False)
		last_nft = self.used_fn  - self.used_fn// self.s_part_num*2
		
		self.s_part3_conv = nn.Conv3d(last_nft ,last_nft , kernel_size=(1,3,3), stride=(1,1,1),dilation=(1,2,2),
							   padding=(0,2,2), bias=False)
		
		self.non_linear = nn.Sequential(
			nn.BatchNorm3d(self.used_fn),
			nn.ReLU(True)
		)
		

		self.cs_l = BatchConv3DLayer(self.used_fn , self.used_fn ,stride=1, padding=0, dilation=1)
		self.max_pool = nn.MaxPool3d(kernel_size=(3,3,3), stride=(1,1,1),
							   padding=(1,1,1))
		self.max_project = nn.Conv3d(self.used_fn , self.used_fn , kernel_size=(1,1,1), stride=(1,1,1),dilation=1,
							   padding=(0,0,0), bias=False)

		self.bn2 = nn.BatchNorm3d(planes)

		self.planes = planes
		p_inner_c = self.used_fn//dyn_network_ratio
		#### implementation of ESP-Net
		self.p_prod1 = nn.Sequential(
			nn.Conv3d(self.used_fn , p_inner_c , kernel_size=(1,1,1), stride=(1,1,1),dilation=1,
							   padding=(0,0,0), bias=False),
			nn.BatchNorm3d(p_inner_c),
			nn.ReLU(inplace=True),
			nn.Conv3d(p_inner_c,p_inner_c,3,1,1),
			nn.BatchNorm3d(p_inner_c),
			nn.ReLU(inplace=True),
			nn.Conv3d(p_inner_c,p_inner_c,3,1,1),
			nn.BatchNorm3d(p_inner_c),
			nn.ReLU(inplace=True),
			nn.Conv3d(p_inner_c,p_inner_c*2,3,2,2),
			nn.BatchNorm3d(p_inner_c*2),
			nn.ReLU(inplace=True),

			nn.Conv3d(p_inner_c*2,p_inner_c*2,3,1,1),
			nn.BatchNorm3d(p_inner_c*2),
			nn.ReLU(inplace=True),
			nn.Conv3d(p_inner_c*2,p_inner_c*2,3,1,1),
			nn.BatchNorm3d(p_inner_c*2),
			nn.ReLU(inplace=True),
			nn.Conv3d(p_inner_c*2,p_inner_c*4,3,2,2),
			nn.BatchNorm3d(p_inner_c*4),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool3d((1,1,1))
		)
		### generating M of size C x C
		self.p_prod2 = nn.Sequential(
			nn.Linear(p_inner_c*4,self.used_fn * self.used_fn),
		)
		

	def forward(self, x):
		assert x.size()[1] == self.planes
		if self.ratio == 1:
			st_feature = x
		else:
			st_feature = self.compress_c1(x)

		b,c,t,h,w = st_feature.size()

		M = self.p_prod2(self.p_prod1(st_feature).squeeze())

		max_x  =self.max_pool(st_feature)
		max_x = self.max_project(max_x)

		part1  = st_feature[:,0 :0 + self.used_fn//self.s_part_num ]
		part2  = st_feature[:,0 + self.used_fn//self.s_part_num  : 0 + self.used_fn//self.s_part_num  *2]
		part3  = st_feature[:,0 + self.used_fn//self.s_part_num*2  :]
		# part4  = st_feature[:,0 + self.used_fn//self.s_part_num*3  : ]

		

		part1 = part1
		part2 = self.t_part2_conv(part2)
		part3 = self.t_part3_conv(part3)
		# part4 = self.t_part4_conv(part4)

		st_ft = torch.cat((part1,part2,part3),dim=1) ## b c t h w
		st_ft = self.non_linear(st_ft)
		b = st_ft.size(0)
		batched_st_ft = st_ft.unsqueeze(1)
		
		M = M.reshape(b,self.used_fn,self.used_fn,1,1,1)

		st_feature = self.cs_l.forward(x=batched_st_ft, weight=M)
		st_feature = st_feature.reshape(b,c,t,h,w)

		part1  = st_feature[:,0 :0 + self.used_fn//self.t_part_num ]
		part2  = st_feature[:,0 + self.used_fn//self.t_part_num  : 0 + self.used_fn//self.t_part_num  *2]
		part3  = st_feature[:,0 + self.used_fn//self.t_part_num*2  :]



		part1 = part1
		part2 = self.s_part2_conv(part2)
		part3 = self.s_part3_conv(part3)
		# part4 = self.s_part4_conv(part4)
		
		st_ft = torch.cat((part1,part2,part3),dim=1)+max_x
		if self.ratio > 1:
			st_ft = self.expand_c1(st_ft)
		if self.is_res:
			out  = st_ft + x
		else:
			out = st_ft

		return out,st_ft,M[:,:,:,0,0,0]



