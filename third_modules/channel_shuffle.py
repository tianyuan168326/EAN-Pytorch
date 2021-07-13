import torch
import torch.nn as nn
import torch.nn.functional as F

def channel_shuffle_3d(x, groups):
    batchsize, num_channels, t, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, t,height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1,t, height, width)

    return x




class ChannelShuffleLayer(nn.Module):
    def __init__(self,g ):
        super(ChannelShuffleLayer, self).__init__()
        self.g = g
    def forward(self, x):
        return channel_shuffle_3d(x,self.g)