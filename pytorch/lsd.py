import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from functools import partial
from collections import OrderedDict

def drop_connect(inputs, p, training):
    """Drop connect.
       
    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.

    Returns:
        output: Output after drop connection.
    """
    assert p >= 0 and p <= 1, 'p must be in range of [0,1]'

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.act(out)
        return out

class Pad(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return F.pad(x, (0,1,0,1), mode='constant', value=0)


class LSDConvBlock(nn.Module):
    def __init__(self, kernel_size=3, input_filters=None, output_filters=None, stride=None, expand_ratio=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.bn_mom = 1-0.99
        self.bn_eps = 0.001

        inp = self.input_filters
        oup = self.input_filters * self.expand_ratio

        #expand
        if self.expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(num_features=oup, momentum=self.bn_mom, eps=self.bn_eps)

        #depthwise
        
        if self.stride == 1:
            self.depthwise_conv = nn.Conv2d(in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
                kernel_size=self.kernel_size, stride=self.stride, padding=int((self.kernel_size-1)/2),bias=False)
        else:
            self.depthwise_conv = nn.Sequential(
                Pad(),
                nn.Conv2d(in_channels=oup, out_channels=oup, groups=oup, 
                          kernel_size=self.kernel_size, stride=self.stride, padding=int((self.kernel_size)/5),bias=False))

        self.bn1 = nn.BatchNorm2d(num_features=oup, momentum=self.bn_mom, eps=self.bn_eps)

        #pointwise
        final_oup = self.output_filters
        self.project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self.bn_mom, eps=self.bn_eps)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = inputs
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
            x = self.bn0(x)
            x = self.act(x)

        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.project_conv(x)
        x = self.bn2(x)
        if self.stride == 1 and self.input_filters == self.output_filters:
            x = drop_connect(x, p=0.2, training=self.training)
            x = x + inputs
        return x

class LSD(nn.Module):
    def __init__(self, max_depth=80):
        super().__init__()
        self.bn_mom = 1-0.99
        self.bn_eps = 0.001
        self.max_depth = max_depth

        #Encoder
        #stem
        in_channels = 3  # rgb
        out_channels = 32

        self.conv_stem = nn.Sequential(
                Pad(),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                          kernel_size=3, stride=2,bias=False))
        self.bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=self.bn_mom, eps=self.bn_eps)


        blocks_args=[
        {'name':'block1', 'kernel_size':3, 'repeat':1, 'input_filters':32,  'output_filters':16,  'stride':1, 'expand_ratio':1},
        {'name':'block2', 'kernel_size':3, 'repeat':2, 'input_filters':16,  'output_filters':24,  'stride':2, 'expand_ratio':6},
        {'name':'block3', 'kernel_size':5, 'repeat':2, 'input_filters':24,  'output_filters':40,  'stride':2, 'expand_ratio':6},
        {'name':'block4', 'kernel_size':3, 'repeat':3, 'input_filters':40,  'output_filters':80,  'stride':2, 'expand_ratio':6},
        {'name':'block5', 'kernel_size':5, 'repeat':3, 'input_filters':80,  'output_filters':112, 'stride':1, 'expand_ratio':6},
        {'name':'block6', 'kernel_size':5, 'repeat':4, 'input_filters':112, 'output_filters':192, 'stride':2, 'expand_ratio':6},
        {'name':'block7', 'kernel_size':3, 'repeat':1, 'input_filters':192, 'output_filters':320, 'stride':1, 'expand_ratio':6}]

        self.blocks = nn.ModuleList([])
        for arg in blocks_args:
            self.blocks.append(LSDConvBlock(kernel_size  =arg['kernel_size'],
                                            input_filters=arg['input_filters'],
                                            output_filters=arg['output_filters'],
                                            stride= arg['stride'],
                                            expand_ratio=arg['expand_ratio']))
            if arg['repeat'] > 1:
                arg['stride'] = 1
                arg['input_filters'] = arg['output_filters']
            for _ in range(arg['repeat'] - 1):
                self.blocks.append(LSDConvBlock(kernel_size  =arg['kernel_size'],
                                            input_filters=arg['input_filters'],
                                            output_filters=arg['output_filters'],
                                            stride= arg['stride'],
                                            expand_ratio=arg['expand_ratio']))
        
        #Decoder
        self.decoder_convs = nn.ModuleList()
        decoder_ch_in = [320, 384, 192, 224, 160, 80, 80, 40, 48, 24, 32]
        decoder_ch_out = [192, 192, 112, 80, 80, 40, 40, 24, 24, 16, 16]
        for i in range(11):
            self.decoder_convs.append(ConvBlock(decoder_ch_in[i], decoder_ch_out[i]))

        self.disp = ConvBlock(16, 1)
        self.act = nn.ReLU()

    def forward(self, inputs, focal=None):
        #Encoder
        #Stem
        x = self.bn0(self.conv_stem(inputs))
        x = self.act(x)

        skip_features = []

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if (idx == 0 or idx == 2 or idx == 4 or idx == 7 or idx == 10 or idx == 14 or idx == 15):
                skip_features.append(x)

        #Decoder
        x = skip_features[-1]
        x = self.decoder_convs[0](x)
        x = [x]
        x += [skip_features[5]]
        x = torch.cat(x, 1)
        x = self.decoder_convs[1](x)

        x = self.decoder_convs[2](x) / self.max_depth
        x = upsample(x)

        x = [x]
        x += [skip_features[4]]
        x = torch.cat(x, 1)
        x = self.decoder_convs[3](x)

        x = [x]
        x += [skip_features[3]]
        x = torch.cat(x, 1)
        x = self.decoder_convs[4](x)

        x = self.decoder_convs[5](x) / self.max_depth
        x = upsample(x)
        # self.d8outputs = self.sigmoid(self.decoder_convs[("dispconv", 3)](x))

        x = [x]
        x += [skip_features[2]]
        x = torch.cat(x, 1)
        x = self.decoder_convs[6](x)

        x = self.decoder_convs[7](x) / self.max_depth
        x = upsample(x)
        # self.d4outputs = self.sigmoid(self.decoder_convs[("dispconv", 2)](x))

        x = [x]
        x += [skip_features[1]]
        x = torch.cat(x, 1)
        x = self.decoder_convs[8](x)

        x = self.decoder_convs[9](x) / self.max_depth
        x = upsample(x)
        # self.d2outputs = self.sigmoid(self.decoder_convs[("dispconv", 1)](x))

        x = [x]
        x += [skip_features[0]]
        x = torch.cat(x, 1)
        x = self.decoder_convs[10](x) / self.max_depth

        x = upsample(x)
        x = self.disp(x)
        return x * self.max_depth

