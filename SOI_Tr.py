"""
An example combining `Temporal Shift Module` with `ResNet`. This implementation
is based on `Temporal Segment Networks`, which merges temporal dimension into
batch, i.e. inputs [N*T, C, H, W]. Here we show the case with residual connections
and zero padding with 8 frames as input.
"""
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch as tr
  
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


class Residual(nn.Module):
	def __init__(self, fn):
		super().__init__()
		self.fn = fn
	def forward(self, x, **kwargs):
		return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
	def __init__(self, dim, fn):
		super().__init__()
		# self.norm = nn.LayerNorm(dim)
		self.norm = nn.BatchNorm1d(dim)
		self.fn = fn
	def forward(self, x, **kwargs):
		b,t,c = x.size()
		x_reshape = x.reshape(b*t,c)
		x_reshape = self.norm(x_reshape)
		x_normed = x_reshape.reshape(b,t,c)
		return self.fn(x_normed, **kwargs)



class FeedForward(nn.Module):
	def __init__(self, dim, hidden_dim, dropout = 0.):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(True),
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(True),
			nn.Linear(hidden_dim, dim),
		)
	def forward(self, x):
		# print("x",x.size())
		b,n,c = x.size()
		x_reshaped = x.reshape(b*n,c)
		x_reshaped =  self.net(x_reshaped)
		x_reshaped = x_reshaped.reshape(b,n,c)
		return x_reshaped

class Attention(nn.Module):
	def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
		super().__init__()
		inner_dim = dim_head *  heads
		self.heads = heads
		self.scale = dim_head ** -0.5

		self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
		self.to_out = nn.Sequential(
			nn.Linear(inner_dim, dim),
			nn.Dropout(dropout)
		)

	def forward(self, x, mask = None):
		b, n, _, h = *x.shape, self.heads
		qkv = self.to_qkv(x).chunk(3, dim = -1)
		q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

		dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
		mask_value = -torch.finfo(dots.dtype).max

		if mask is not None:
			mask = F.pad(mask.flatten(1), (1, 0), value = True)
			assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
			mask = mask[:, None, :] * mask[:, :, None]
			dots.masked_fill_(~mask, mask_value)
			del mask

		attn = dots.softmax(dim=-1)

		out = torch.einsum('bhij,bhjd->bhid', attn, v)
		out = rearrange(out, 'b h n d -> b n (h d)')
		out =  self.to_out(out)
		return out

class Transformer(nn.Module):
	def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
		super().__init__()
		self.layers = nn.ModuleList([])
		for _ in range(depth):
			self.layers.append(nn.ModuleList([
				Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
				Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
			]))
	def forward(self, x, mask = None):
		for attn, ff in self.layers:
			x = attn(x, mask = mask)
			x = ff(x)
		return x

MAX_NUM = 4


class SparseTransformer(nn.Module):
	expansion = 1

	def __init__(self,input_c = 2048, down_ratio = 4, obj_down_ratio = 4):
		super(SparseTransformer, self).__init__()
		ft_c = input_c//down_ratio
		obj_ft_c =ft_c//obj_down_ratio
		self.squeeze_l = nn.Conv3d(input_c,ft_c,1,1,0,bias=False)
		self.expand_l = nn.Linear(ft_c,input_c,bias=False)
		self.score_predictor = nn.Sequential(
			nn.Conv3d(ft_c,obj_ft_c,1,1,0),
			nn.ReLU(inplace=True),
			nn.BatchNorm3d(obj_ft_c),

			nn.Conv3d(obj_ft_c,obj_ft_c,3,1,1),
			nn.ReLU(inplace=True),
			nn.BatchNorm3d(obj_ft_c),
			nn.Conv3d(obj_ft_c,obj_ft_c,(1,3,3),1,(0,1,1)),
			nn.ReLU(inplace=True),
			nn.BatchNorm3d(obj_ft_c),

			nn.Conv3d(obj_ft_c,MAX_NUM,(1,3,3),1,(0,1,1)),
		)
		patch_num = 8*7*7
		# self.pos_embedding = 
		# self.register_parameter("pos_embedding",nn.Parameter(torch.randn()))
		self.register_parameter("pos_embedding",torch.nn.Parameter(torch.FloatTensor(ft_c,patch_num)))
		torch.nn.init.normal(self.pos_embedding)
		# self.add_module("pos_embedding",self.pos_embedding)
		# self.pos_emb = nn.Linear(1,input_c)
		self.transformer = Transformer(ft_c,2,8,ft_c//8,ft_c,0)
		self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))
		self.Tr_BN = nn.BatchNorm1d(input_c)
		nn.init.constant_(self.Tr_BN.weight, 0)
		nn.init.constant_(self.Tr_BN.bias, 0)
	def forward(self, x):
		x_avg  = self.avg_pool(x).squeeze()
		x = self.squeeze_l(x)
		b,c,t,h,w = x.size()
		x_score = self.score_predictor(x) ## b m t h w
		x_score = F.softmax(x_score.reshape(b,MAX_NUM,t,-1),dim=-1).\
			reshape(b,MAX_NUM,t,h,w)

		
		max_loc_emb = self.pos_embedding.unsqueeze(0).repeat(b,1,1)
	
		x_reshaped_pos_agged = x.reshape(b,c,t*h*w)
		x_selected = x_reshaped_pos_agged
		x_selected = x_selected+max_loc_emb
		x_selected = x_selected.reshape(b,c,t,h,w)
		x_selected = x_selected.unsqueeze(2).repeat(1,1,MAX_NUM,1,1,1)\
			* x_score.unsqueeze(1).repeat(1,c,1,1,1,1)
		x_selected = x_selected.sum(-1).sum(-1) ## b c ,m ,t
		x_selected = x_selected.reshape(b,c,-1)

		x_selected = F.dropout(x_selected,0)
		x_selected = x_selected.transpose(1,2) ## b n c

		x_output = self.transformer(x_selected)

		out = x_output.mean(1) ## b c
		out = self.expand_l(out)
		out = self.Tr_BN(out)+ x_avg
		return out,x_score    
