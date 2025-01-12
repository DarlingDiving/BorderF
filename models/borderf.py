import torch 
import torch.nn as nn
import torch.nn.functional as F
from .basicmodule import BasicModule
import torch_dct as dct

'''
feature extraction block
'''

class Laplace(nn.Module):
    def __init__(self):
        super(Laplace, self).__init__()

        kernel_2 = [[-1, -1, -1],
                  [-1, 8, -1],
                  [-1, -1, -1]]

        kernel = torch.FloatTensor(kernel_2).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=1)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=1)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=1)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


'''
ResNeXt
'''

class ResNeXtBottleneck(BasicModule):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.leaky_relu(residual + bottleneck, inplace=True)

class ResNeXt(BasicModule):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, opt, in_channels, spp_level=3):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt, self).__init__()
        self.spp_level = spp_level
        self.num_grids = 0
        for i in range(spp_level):
            self.num_grids += 2**(i*2)
        self.cardinality = opt.cardinality
        self.depth = opt.depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = opt.base_width
        self.widen_factor = opt.widen_factor
        self.nlabels = opt.nlabels
        # self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor, 512 * self.widen_factor]
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn_1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        # self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        # self.stage_4 = self.block('stage_4', self.stages[3], self.stages[4], 2)
        self.classifier = nn.Linear(self.stages[len(self.stages)-1]*2, opt.nlabels)
        
        # self.spp_layer = SPPLayer(spp_level)
        # self.spp_tail = nn.Sequential(OrderedDict([
        #     ('fc1', nn.Linear(self.num_grids*1024,1024)),
        #     ('fc1_relu', nn.ReLU()),
        #     ('fc2', nn.Linear(1024,2)),
        # ]))

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))

        # init.kaiming_normal(self.classifier.weight)
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                # if 'conv' in key:
                    # init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.base_width, self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width,
                                                   self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = self.bn_1.forward(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.maxpool.forward(x)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)

        x = self.adaptive_pool(x)
        return x

'''
proposed network
'''

class BorderF(BasicModule):
    def __init__(self,opt):
        super().__init__()
        self.feature0 = ResNeXt(opt,3)  #for raw img
        self.feature = ResNeXt(opt,3)   #for laplace img
        self.flatten=torch.nn.Flatten()
        self.laplace = Laplace()
        self.classifier = ResNeXt(opt,3+3).classifier
    def forward(self,x):
        feature = self.laplace(x)  
        f_rgb=self.feature0(dct.dct_2d(x))
        f_laplace = self.feature(feature)

        f = torch.cat((f_rgb,f_laplace),dim=1)
        f=self.flatten(f)
        out = self.classifier(f)
        return out,f
