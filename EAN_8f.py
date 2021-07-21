"""
An example combining `Temporal Shift Module` with `ResNet`. This implementation
is based on `Temporal Segment Networks`, which merges temporal dimension into
batch, i.e. inputs [N*T, C, H, W]. Here we show the case with residual connections
and zero padding with 8 frames as input.
"""
import sys
sys.path.append("/data/tianyuan/AST/checkpoints/EAB_SOITr/ablation_res50/SOTA_8frame_lite")
import torch.nn as nn
import torch.nn.functional as F
import torch as tr
# from tsm_util import tsm
import torch.utils.model_zoo as model_zoo
# from spatial_correlation_sampler import SpatialCorrelationSampler
from EAB_rfs555 import *
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
		   'resnet152']


model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
	"""1x1x1 convolution"""
	return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

class BasicBlock2(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, num_segments,stride=1, downsample=None, remainder=0):
		super(BasicBlock2, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride
		self.remainder =remainder
		self.num_segments = num_segments        

	def forward(self, x):
		identity = x  
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		
		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out        
	
class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, num_segments, stride=1, downsample=None, remainder=0):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride
		self.remainder= remainder
		self.num_segments = num_segments

	def d2_to_d3(self,x):
		bt,c,h,w = x.size()
		t = 16
		b = bt//t
		return x.reshape(b,t,c,h,w).transpose(1,2)
	def d3_to_d2(self,x):
		b,c,t,h,w = x.size()
		return x.transpose(1,2).reshape(b*t,c,h,w)
	def forward(self, x):
		identity = x  
		# out = tsm(x, self.num_segments, 'zero') 
		out = x
		out = self.conv1(out)
		out = self.bn1(out)
		out = self.relu(out)
	   
		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class ShiftModule(nn.Module):
    def __init__(self, input_channels, n_segment=8,n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        self.input_channels = input_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(self.fold_div*self.fold, self.fold_div*self.fold,
                kernel_size=3, padding=1, groups=self.fold_div*self.fold,
                bias=False)

        if mode == 'shift':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1 # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
            if 2*self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute(0, 3, 4, 2, 1) # (n_batch, h, w, c, n_segment)
        x = x.contiguous().view(n_batch*h*w, c, self.n_segment)
        x = self.conv(x) # (n_batch*h*w, c, n_segment)
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute(0, 4, 3, 1, 2) # (n_batch, n_segment, c, h, w)
        x = x.contiguous().view(nt, c, h, w)
        return x


class MEModule(nn.Module):
    """ Motion exciation module
    
    :param reduction=16
    :param n_segment=8/16
    """
    def __init__(self, channel, reduction=16, n_segment=8):
        super(MEModule, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.n_segment = n_segment
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel//self.reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel//self.reduction)

        self.conv2 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel//self.reduction,
            kernel_size=3,
            padding=1,
            groups=channel//self.reduction,
            bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)

        self.conv3 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)

        self.identity = nn.Identity()

    def forward(self, x):
        nt, c, h, w = x.size()
        bottleneck = self.conv1(x) # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck) # nt, c//r, h, w

        # t feature
        reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])  # n, t, c//r, h, w
        t_fea, __ = reshape_bottleneck.split([self.n_segment-1, 1], dim=1) # n, t-1, c//r, h, w

        # apply transformation conv to t+1 feature
        conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
        # reshape fea: n, t, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment) + conv_bottleneck.size()[1:])
        __, tPlusone_fea = reshape_conv_bottleneck.split([1, self.n_segment-1], dim=1)  # n, t-1, c//r, h, w
        
        # motion fea = t+1_fea - t_fea
        # pad the last timestamp
        diff_fea = tPlusone_fea - t_fea # n, t-1, c//r, h, w
        # pad = (0,0,0,0,0,0,0,1)
        diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0)  # n, t, c//r, h, w
        diff_fea_pluszero = diff_fea_pluszero.view((-1,) + diff_fea_pluszero.size()[2:])  #nt, c//r, h, w
        y = self.avg_pool(diff_fea_pluszero)  # nt, c//r, 1, 1
        y = self.conv3(y)  # nt, c, 1, 1
        y = self.bn3(y)  # nt, c, 1, 1
        y = self.sigmoid(y)  # nt, c, 1, 1
        y = y - 0.5
        output = x + x * y.expand_as(x)
        return output

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes,num_segments, stride=1, downsample=None, remainder=0):
		super(Bottleneck, self).__init__()
		self.inplanes = inplanes
		self.conv1 = conv1x1(inplanes, planes)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes, stride)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = conv1x1(planes, planes * self.expansion)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride
		self.remainder= remainder        
		self.num_segments = num_segments
		# self.t_conv = ShiftModule(planes, n_segment=8, n_div=2, mode='fixed')  
		self.t_conv = ShiftModule(planes, n_segment=self.num_segments, n_div=8, mode='shift')   
		self.me = MEModule(planes, reduction=16, n_segment=8)

	def forward(self, x):
		identity = x  
		out = x
		out = self.conv1(out)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.me(out)
		# if self.inplanes>64:
		# 	out = self.t_conv(out)
		out = self.t_conv(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out



from SOI_Tr import SparseTransformer	

class ResNet(nn.Module):

	def __init__(self, block, block2, layers, num_segments, flow_estimation, num_classes=1000, zero_init_residual=False):
		super(ResNet, self).__init__()
		self.inplanes = 64
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()          
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.softmax = nn.Softmax(dim=1)        
		self.num_segments = num_segments     

	   
		self.layer1 = self._make_layer(block, 64, layers[0], num_segments=num_segments)
		self.layer2 = self._make_layer(block, 128, layers[1],  num_segments=num_segments, stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2],  num_segments=num_segments, stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3],  num_segments=num_segments, stride=2)       
		self.relation1 = STCeptionLayer_max(64,4)
		self.relation2 = STCeptionLayer_max(64*block.expansion,4)
		self.relation3 = STCeptionLayer_max(128*block.expansion,4)
		self.relation4 = STCeptionLayer_max(256*block.expansion,4)
		self.sparse_transformer = SparseTransformer(2048,8,4,4)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)                   



		self.x_down = 2
		self.p_num=7
		self.p1 = self.p2 = 112//self.p_num
		self.patch_rgb_dim = self.p1 * self.p2 *3
		self.patch_dim = 256
		self.stm_proj1 = nn.Sequential(
			nn.Conv3d(self.patch_rgb_dim,self.patch_dim,1,1,0,bias=False),
		)

		self.stm_proj2 = nn.Sequential(
			nn.Conv3d(self.patch_dim*2,self.patch_dim,1,1,0,bias=False),
			nn.BatchNorm3d(self.patch_dim),
		)
		self.stm_st_model1 = nn.Sequential(
			nn.Conv3d(self.patch_dim,self.patch_dim,3,1,1,bias=False,groups = 16),
			nn.BatchNorm3d(self.patch_dim),
			nn.ReLU(inplace=True),
			nn.Conv3d(self.patch_dim,self.patch_dim,1,1,0,bias=False),
			nn.BatchNorm3d(self.patch_dim),
		)

		self.stm_st_model2 = nn.Sequential(
			nn.Conv3d(self.patch_dim,self.patch_dim,3,1,1,bias=False,groups =16),
			nn.BatchNorm3d(self.patch_dim),
		)

		self.proj_back = nn.Sequential(
			nn.Conv3d(self.patch_dim,8* 8 *(16),1,1,0,bias=False),
		)
		self.proj_back[0].weight.data = self.proj_back[0].weight.data*0.01
		
		for n,m in self.named_modules():  
			if "expand_c1"  in n:
				continue
			elif "Tr_BN"  in n:
				continue
			elif "t_conv"  in n:
				continue
			elif "proj_back" in n:
				continue
		
			elif isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.Conv3d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm1d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm3d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	
	def _make_layer(self, block, planes, blocks, num_segments, stride=1):       
		downsample = None        
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)
			
		layers = []
		layers.append(block(self.inplanes, planes, num_segments, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			remainder =int( i % 3)
			layers.append(block(self.inplanes, planes, num_segments, remainder=remainder))
			
		return nn.Sequential(*layers)            
	
	
	def d2_to_d3(self,x,t=8):
		bt,c,h,w = x.size()
		# t = 8
		b = bt//t
		return x.reshape(b,t,c,h,w).transpose(1,2)
	def d3_to_d2(self,x):
		b,c,t,h,w = x.size()
		return x.transpose(1,2).reshape(b*t,c,h,w)
	def forward(self, x):
		x_t = self.d2_to_d3(x,t=40) ### b c 40 h w
		b,c,l,h,w = x_t.size()
		x_flatten = x_t.reshape(b,c,8,5,h,w)
		x = x_flatten[:,:,:,2] ## mid frames

		x_short_clip = x_flatten.permute(0,2,1,3,4,5).reshape(b*8,c,5,h,w)
		x_short_clip_down_sample = F.upsample(x_short_clip,scale_factor= (1,0.5,0.5))
		x_short_clip_down_sample = x_short_clip_down_sample[:,:,1:5] - x_short_clip_down_sample[:,:,0:4]
		X_patches = rearrange(x_short_clip_down_sample,"b c t (h p1) (w p2) -> b (p1 p2 c) t h w",p1 = self.p1,p2 = self.p2)
		x_features = self.stm_proj1(X_patches)
		
		x_features = x_features+self.stm_st_model1(x_features)
		x_features = x_features+self.stm_st_model2(x_features)
		x_patches = self.proj_back(x_features)
		X_patches = rearrange(x_patches,"b (p1 p2 c) t h w -> b c t (h p1) (w p2)",c = 16,p1 = 8,p2 = 8)
		x_res = X_patches.reshape(-1,8,4*16,56,56).transpose(1,2)

		x = self.d3_to_d2(x)
		input = x

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.d2_to_d3(x)
		x = x+ x_res
		x,_,matrix1 = self.relation1(x)
		x = self.d3_to_d2(x)

		x = self.layer1(x)   
		x = self.d2_to_d3(x)
		x,_,matrix2 = self.relation2(x)
		x = self.d3_to_d2(x)                       
		x = self.layer2(x) 
		x = self.d2_to_d3(x)
		x,_,matrix3 = self.relation3(x)
		x = self.d3_to_d2(x)       
		
		
		x = self.layer3(x)  
		x = self.d2_to_d3(x)
		x,_,matrix4 = self.relation4(x)
		x = self.d3_to_d2(x) 

		x = self.layer4(x)
		x_d3 = self.d2_to_d3(x)
		# feature_from_transformer = x_d3.mean(-1).mean(-1).mean(-1)

		feature_from_transformer,x_score = self.sparse_transformer(x_d3)
		
		x_out = feature_from_transformer 
		# print(feature_from_transformer.size(),x.size(),"fuck")		   
		x = self.fc(x_out)      
		return x, {
			'matrix1':matrix1,
			'matrix2':matrix2,
			'matrix3':matrix3,
			'matrix4':matrix4,
			'input':self.d2_to_d3(input)
		}


def resnet18(pretrained=True, shift='TSM',num_segments = 8, flow_estimation=0,alpha=0,beta=0, **kwargs):
	"""Constructs a ResNet-18 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	if (shift =='TSM'):    
		model = ResNet(BasicBlock, BasicBlock, [2, 2, 2, 2], num_segments=num_segments , flow_estimation=flow_estimation, **kwargs)  
	if pretrained:
		pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
		new_state_dict =  model.state_dict()
		for k, v in pretrained_dict.items():
			if (k in new_state_dict):
				new_state_dict.update({k:v})      
#                 print ("%s layer has pretrained weights" % k)
		model.load_state_dict(new_state_dict)
	return model


def resnet34(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=0,**kwargs):
	"""Constructs a ResNet-34 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	if (shift =='TSM'):
		model = ResNet(BasicBlock, BasicBlock, [3, 4, 6, 3],num_segments=num_segments , flow_estimation=flow_estimation,  **kwargs)        
	if pretrained:
		pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
		new_state_dict =  model.state_dict()
		for k, v in pretrained_dict.items():
			if (k in new_state_dict):
				new_state_dict.update({k:v})      
#                 print ("%s layer has pretrained weights" % k)
		model.load_state_dict(new_state_dict)
	return model


def resnet50(pretrained=True, shift='TSM',num_segments = 8, flow_estimation=0,alpha=0,beta=0, **kwargs):
	"""Constructs a ResNet-50 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	if (shift =='TSM'):    
		model = ResNet(Bottleneck, Bottleneck, [3, 4, 6, 3],num_segments=num_segments , flow_estimation=flow_estimation, **kwargs)          
	if pretrained:
		pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
		new_state_dict =  model.state_dict()
		for k, v in pretrained_dict.items():
			if (k in new_state_dict):
				new_state_dict.update({k:v})      
#                 print ("%s layer has pretrained weights" % k)
		model.load_state_dict(new_state_dict)
	return model


def resnet101(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=0, **kwargs):
	"""Constructs a ResNet-101 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	if (shift =='TSM'):    
		model = ResNet(Bottleneck, Bottleneck, [3, 4, 23, 3],num_segments=num_segments , flow_estimation=flow_estimation, **kwargs)          
	if pretrained:
		pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
		new_state_dict =  model.state_dict()
		for k, v in pretrained_dict.items():
			if (k in new_state_dict):
				new_state_dict.update({k:v})      
#                 print ("%s layer has pretrained weights" % k)
		model.load_state_dict(new_state_dict)
	return model