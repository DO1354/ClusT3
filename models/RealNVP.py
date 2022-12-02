import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm

class WeightNormConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=0, bias=True):
        super(WeightNormConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias))

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, bottleneck):
        '''
        :param dim: input dimension (channels)
        :param bottleneck: Whether to use a bottleneck block or not
        '''
        super(ResidualBlock, self).__init__()
        self.in_block = nn.Sequential(nn.BatchNorm2d(dim),
                                      nn.ReLU())
        if bottleneck:
            self.block = nn.Sequential(WeightNormConv2d(dim, dim, 1),
                                       nn.BatchNorm2d(dim),
                                       nn.ReLU(),
                                       WeightNormConv2d(dim, dim, 3, padding=1),
                                       nn.BatchNorm2d(dim),
                                       nn.ReLU(),
                                       WeightNormConv2d(dim, dim, 1))
        else:
            self.block = nn.Sequential(WeightNormConv2d(dim, dim, 3, passinf=1),
                                       nn.BatchNorm2d(dim),
                                       nn.ReLU(),
                                       WeightNormConv2d(dim, dim, 3, padding=1))

    def forward(self, x):
        return x + self.block(self.in_block(x))

class ResModule(nn.Module):
    def __init__(self, in_dim, dim, out_dim, res_blocks, bottleneck):
        '''
        :param in_dim: Number of input features
        :param dim: Number of features in residual blocks
        :param out_dim: Number of output features (should be double of input)
        :param res_blocks: Number of residual blocks
        :param bottleneck: Whether to use bottleneck block or not
        '''
        super(ResModule,self).__init__()
        self.res_blocks = res_blocks
        self.in_block = WeightNormConv2d(in_dim, dim, padding=1)
        self.core_block = nn.ModuleList([ResidualBlock(dim, bottleneck) for _ in range(res_blocks)])
        self.out_block = nn.Sequential(nn.BatchNorm2d(dim),
                                       nn.ReLU(),
                                       WeightNormConv2d(dim, out_dim, 1))

    def forward(self, x):
        x = self.in_block(x)
        for block in self.core_block:
            x = block(x)
        x = self.out_block(x)
        return x

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

'''Constructor funtion'''
def create_subnet(dims_in, dims_out):
    '''
    :param dims_in: number of input channels
    :param dims_out: number of output channels
    :return: A network that compute an output of "dims_out" size.
    '''
    return ResModule(in_dim=dims_in, dim=dims_in, out_dim=dims_out, res_blocks=1, bottleneck=True)

'''RealNVP'''
def realnvp(channels, resolution, K=3):
    '''
    :param resolution: input spatial resolution (hight/width)
    :param channels: input number of channels
    :param K: number of coupling layers
    :return: flow model
    '''
    inn = Ff.SequenceINN(channels, resolution, resolution)
    for k in range(K):
        #AllInOneBlock already includes activation function and normalization (we don't need to add it)
        inn.append(Fm.AllInOneBlock, subnet_constructor=create_subnet)

    inn.apply(initialize_weights)
    '''with torch.no_grad():
        inn.module_list[-1]'''


    return inn