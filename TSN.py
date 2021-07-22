import torch.nn as nn
from transforms import *
from torch.nn.init import normal_, constant_
import torch.nn.functional as F
import datetime
import numpy as np
import os
class TemporalModel(nn.Module):
	def __init__(self,ft_action,input_size, num_class, num_segments, model, backbone, alpha, beta, 
				dropout = 0.3, target_transforms = None):	
		super(TemporalModel, self).__init__()
		self.input_size = input_size
		self.ft_action  =ft_action
		self.num_segments = num_segments
		self.model = model
		self.backbone = backbone
		self.dropout = dropout
		self.alpha = alpha
		self.beta = beta
		self.target_transforms = target_transforms
		self._prepare_base_model(model,backbone)
		std = 0.001
		if self.dropout == 0:
			setattr(self.base_model,self.base_model.last_layer_name,nn.Linear(self.feature_dim,num_class))
			self.new_fc = None
			normal_(getattr(self.base_model,self.base_model.last_layer_name).weight, 0, std)
			constant_(getattr(self.base_model,self.base_model.last_layer_name).bias, 0)
		else:
			print("self.base_model.last_layer_name",self.base_model.last_layer_name)
			setattr(self.base_model,self.base_model.last_layer_name,nn.Dropout(p=self.dropout))
			self.new_fc = nn.Linear(2048,num_class)
			normal_(self.new_fc.weight, 0, std)
			constant_(self.new_fc.bias, 0)
		
		self.run_i = 0

		

	def _prepare_base_model(self, model, backbone):
		
		import importlib
		lib = importlib.import_module(model)
		self.base_model = getattr(lib,backbone)(num_segments =self.num_segments ,alpha = self.alpha, beta = self.beta)
			# raise ValueError('Unknown model: {}'.format(model))
		
		if 'resnet' in backbone:
			self.base_model.last_layer_name = 'fc'
			self.init_crop_size = self.input_size*1.143
			self.input_mean = [0.485, 0.456, 0.406]
			self.input_std = [0.229, 0.224, 0.225]
			if backbone == 'resnet18' or backbone == 'resnet34':
				self.feature_dim = 512
			else:
				self.feature_dim = 2048
		
		else:
			raise ValueError('Unknown base model: {}'.format(base_model))
	def norm_video(self,v):
		b,c,t,h,w = v.size()
		mean  = torch.tensor([0.485, 0.456, 0.406]).cuda(v.get_device())
		std  = torch.tensor([0.229, 0.224, 0.225]).cuda(v.get_device())
		mean  =mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		std  =std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		mean = mean.repeat(b,1,t,h,w)
		std = std.repeat(b,1,t,h,w)
		return (v-std)/mean
	def forward(self, input,fixed_weight = False):
		input = input.view((-1, self.num_segments,3) + input.size()[-2:])
		src_input = input.transpose(1,2).contiguous()
		
		if True:
			b,c,t,h,w = src_input.size()### srouces
			
			# video_input = self.norm_video(src_input)
			video_input = src_input
			video_input = video_input.transpose(1,2).reshape(b*t,c,h,w)
			base_out,vis = self.base_model(video_input)
			base_out = base_out.squeeze()
			if self.dropout > 0:
				base_out1 = self.new_fc(base_out)
		
		return base_out1,vis

	
	@property
	def crop_size(self):
		return self.input_size

	@property
	def scale_size(self):
		return self.input_size * self.init_crop_size // self.input_size

	def get_augmentation(self):
		if "something" in os.environ["ds"]:
			return torchvision.transforms.Compose([GroupScale(int(self.scale_size)),
													GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(self.target_transforms)])
		else:
			return torchvision.transforms.Compose([
													GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(self.target_transforms)])
	 