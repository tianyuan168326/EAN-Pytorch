import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchConv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 padding=0, dilation=1):
        super(BatchConv2DLayer, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, weight, bias=None):
        if bias is None:
            assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
        else:
            assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
                0], "dim=0 of bias must be equal in size to dim=0 of weight"

        b_i, b_j, c, h, w = x.shape
        b_i, out_channels, in_channels, kernel_height_size, kernel_width_size = weight.shape

        out = x.permute([1, 0, 2, 3, 4]).contiguous().view(b_j, b_i * c, h, w)
        weight = weight.view(b_i * out_channels, in_channels, kernel_height_size, kernel_width_size)

        out = F.conv2d(out, weight=weight, bias=None, stride=self.stride, dilation=self.dilation, groups=b_i,
                       padding=self.padding)


        out = out.view(b_j, b_i, out_channels, out.shape[-2], out.shape[-1])

        out = out.permute([1, 0, 2, 3, 4])

        if bias is not None:
            out = out + bias.unsqueeze(1).unsqueeze(3).unsqueeze(3)

        return out

class BatchConv3DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 padding=0, dilation=1):
        super(BatchConv3DLayer, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, weight, bias=None):
        if bias is None:
            assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
        else:
            assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
                0], "dim=0 of bias must be equal in size to dim=0 of weight"

        b_i, b_j, c,t, h, w = x.shape
        b_i, out_channels, in_channels,kernel_t_size, kernel_height_size, kernel_width_size = weight.shape

        out = x.permute([1, 0, 2, 3, 4,5]).contiguous().view(b_j, b_i * c,t, h, w)
        weight = weight.view(b_i * out_channels, in_channels,kernel_t_size, kernel_height_size, kernel_width_size)

        out = F.conv3d(out, weight=weight, bias=None, stride=self.stride, dilation=self.dilation, groups=b_i,
                       padding=self.padding)


        out = out.view(b_j, b_i, out_channels,out.shape[-3], out.shape[-2], out.shape[-1])

        out = out.permute([1, 0, 2, 3, 4,5])

        if bias is not None:
            out = out + bias.unsqueeze(1).unsqueeze(3).unsqueeze(3).unsqueeze(3)

        return out


class BatchConv1DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 padding=0, dilation=1):
        super(BatchConv1DLayer, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, weight, bias=None):
        if bias is None:
            assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
        else:
            assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
                0], "dim=0 of bias must be equal in size to dim=0 of weight"

        b_i, b_j, c, h = x.shape
        b_i, out_channels, in_channels, kernel_width_size = weight.shape

        out = x.permute([1, 0, 2, 3]).contiguous().view(b_j, b_i * c, h)
        weight = weight.view(b_i * out_channels, in_channels, kernel_width_size)

        out = F.conv1d(out, weight=weight, bias=None, stride=self.stride, dilation=self.dilation, groups=b_i,
                       padding=self.padding)

        out = out.view(b_j, b_i, out_channels, out.shape[-1])

        out = out.permute([1, 0, 2, 3])

        if bias is not None:
            out = out + bias.unsqueeze(1).unsqueeze(3)

        return out

class BatchLinearLayer(nn.Module):
    def __init__(self):
        super(BatchLinearLayer, self).__init__()

    def forward(self, x, weight, bias=None):
        if bias is None:
            assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
        else:
            assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
                0], "dim=0 of bias must be equal in size to dim=0 of weight"

        out = torch.bmm(x, weight)

        if bias is not None:
            out = out + bias.unsqueeze(1)

        return out
