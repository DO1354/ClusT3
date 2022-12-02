import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import numpy as np

class Preprocess(nn.Module):
    def __init__(self):
        super(Preprocess, self).__init__()

    def forward(self, x, reverse=False):
        device = x.device
        if reverse:
            return (x.sigmoid() - 0.05) / 0.9
        x += Uniform(0.0, 1.0).sample(x.size()).to(device)
        x = 0.05 + 0.9 * (x / 4.0)
        z = torch.log(x) - torch.log(1-x)
        log_det_jacobian = -x.log() - (1-x).log() + torch.tensor(0.9/4).log().to(device)
        return z, log_det_jacobian

class WeightNormConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super(WeightNormConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(in_channel,
                                                   out_channel,
                                                   kernel_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   bias=bias))

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
        self.in_block = WeightNormConv2d(in_dim, dim, 3, padding=1)
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


class MaskQuadrant:
    def __init__(self, input_quadrant, output_quadrant):
        self.input_quadrant = input_quadrant
        self.output_quadrant = output_quadrant

    @staticmethod
    def _get_quadrant_mask(quadrant, c, h, w):
        b = torch.zeros((1, c, h, w), dtype=torch.float)
        split_h = h // 2
        split_w = w // 2
        if quadrant == 0:
            b[:, :, :split_h, :split_w] = 1.
        elif quadrant == 1:
            b[:, :, :split_h, split_w:] = 1.
        elif quadrant == 2:
            b[:, :, split_h:, split_w:] = 1.
        elif quadrant == 3:
            b[:, :, split_h:, :split_w] = 1.
        else:
            raise ValueError("Incorrect mask quadrant")
        return b

    def mask(self, x):
        # x.shape = (bs, c, h, w)
        c, h, w = x.size(1), x.size(2), x.size(3)
        self.b_in = self._get_quadrant_mask(self.input_quadrant, c, h, w).to(x.device)
        self.b_out = self._get_quadrant_mask(self.output_quadrant, c, h, w).to(x.device)
        x_id = x * self.b_in
        x_change = x * (1 - self.b_in)
        return x_id, x_change

    def forward(self, x):
        self.mask = self.mask.to(x.device)
        x_masked = x * self.mask
        '''scale, shift = self.net(x_masked).chunk(2, dim=1)
        scale = scale.tanh() * self.scale_scale + self.shift_scale
        scale = scale * (1 - self.mask)
        shift = shift * (1 - self.mask)
        x = x * scale + shift
        log_scale = torch.sigmoid(scale).log()'''
        log_scale, shift = self.net(x_masked).chunk(2, dim=1)
        log_scale = log_scale.tanh()*self.scale_scale + self.shift_scale
        log_scale = log_scale*(1 - self.mask)
        shift = shift * (1 - self.mask)
        x = x * log_scale.exp() + shift
        return x, log_scale

    def unmask(self, x_id, x_change):
        return x_id * self.b_in + x_change * (1 - self.b_in)

    def mask_st_output(self, s, t):
        return s * self.b_out, t * self.b_out


class CouplingLayerBase(nn.Module):
    """Coupling layer base class in RealNVP.

    must define self.mask, self.st_net, self.rescale
    """

    def _get_st(self, x):
        # positional encoding
        # bs, c, h, w = x.shape
        # y_coords = torch.arange(h).float().cuda() / h
        # y_coords = y_coords[None, None, :, None].repeat((bs, 1, 1, w))
        # x_coords = torch.arange(w).float().cuda() / w
        # x_coords = x_coords[None, None, None, :].repeat((bs, 1, h, 1))
        # x = torch.cat([x, y_coords, x_coords], dim=1)

        x_id, x_change = self.mask.mask(x)
        st = self.st_net(x_id)
        # st = self.st_net(F.dropout(x_id, training=self.training, p=0.5))
        # st = self.st_net(F.dropout(x_id, training=True, p=0.9))
        s, t = st.chunk(2, dim=1)
        s = self.rescale(torch.tanh(s))

        # positional encoding
        # s = s[:, :-2]
        # t = t[:, :-2]
        # x_id = x_id[:, :-2]
        # x_change = x_change[:, :-2]
        return s, t, x_id, x_change

    def forward(self, x, sldj=None, reverse=True):
        s, t, x_id, x_change = self._get_st(x)
        s, t = self.mask.mask_st_output(s, t)
        # positional encoding
        # s = s[:, :-2]
        # t = t[:, :-2]

        exp_s = s.exp()
        if torch.isnan(exp_s).any():
            raise RuntimeError('Scale factor has NaN entries')
        x_change = (x_change + t) * exp_s
        self._logdet = s.view(s.size(0), -1).sum(-1)
        x = self.mask.unmask(x_id, x_change)
        # positional encoding
        # x = x[:, :-2]
        return x, self._logdet

    def inverse(self, y):
        s, t, x_id, x_change = self._get_st(y)
        s, t = self.mask.mask_st_output(s, t)
        exp_s = s.exp()
        inv_exp_s = s.mul(-1).exp()
        if torch.isnan(inv_exp_s).any():
            raise RuntimeError('Scale factor has NaN entries')
        x_change = x_change * inv_exp_s - t
        self._logdet = -s.view(s.size(0), -1).sum(-1)
        x = self.mask.unmask(x_id, x_change)

        # positional encoding
        # x = x[:, :-2]
        return x

    def logdet(self):
        return self._logdet


class CouplingLayer(CouplingLayerBase):
    """Coupling layer in RealNVP for image data.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        mask (MaskChannelWise or MaskChannelWise): mask.
    """

    def __init__(self, in_dim, dim, out_dim, res_blocks, bottleneck, mask):
        #in_channels, mid_channels, num_blocks, mask, init_zeros=False, use_batch_norm=True, skip=True):
        super(CouplingLayer, self).__init__()

        self.mask = mask

        # self.st_net = ResNet(in_channels, mid_channels, 2*in_channels,
        #                      num_blocks=num_blocks, kernel_size=3, padding=1,
        #                      double_after_norm=False,
        #                      init_zeros=init_zeros, use_batch_norm=use_batch_norm, skip=skip)

        self.st_net = ResModule(in_dim=in_dim, dim=dim, out_dim=out_dim, res_blocks=res_blocks, bottleneck=bottleneck)

        # Learnable scale for s
        self.rescale = nn.utils.weight_norm(Rescale(in_dim))


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x


class AffineCheckerboardTransform(nn.Module):
    def __init__(self, in_dim, dim, out_dim, res_blocks, bottleneck, size, config):
        '''
        :param size: mask height/width
        :param config: 1 for first position, 0 for first position
        '''
        super(AffineCheckerboardTransform, self).__init__()
        self.mask = self.create_mask(size, config)
        self.scale_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.shift_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.net = ResModule(in_dim=in_dim, dim=dim, out_dim=out_dim, res_blocks=res_blocks, bottleneck=bottleneck)

    def create_mask(self, size, config):
        mask = (torch.arange(size).view(-1,1) + torch.arange(size))
        if config == 1:
            mask += 1
        return (mask%2).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        self.mask = self.mask.to(x.device)
        x_masked = x * self.mask
        '''scale, shift = self.net(x_masked).chunk(2, dim=1)
        scale = scale.tanh() * self.scale_scale + self.shift_scale
        scale = scale * (1 - self.mask)
        shift = shift * (1 - self.mask)
        x = x * scale + shift
        log_scale = torch.sigmoid(scale).log()'''
        log_scale, shift = self.net(x_masked).chunk(2, dim=1)
        log_scale = log_scale.tanh()*self.scale_scale + self.shift_scale
        log_scale = log_scale*(1 - self.mask)
        shift = shift * (1 - self.mask)
        x = x * log_scale.exp() + shift
        return x, log_scale

class AffineChannelwiseTransform(nn.Module):
    def __init__(self, in_dim, dim, out_dim, res_blocks, bottleneck, top_half_as_input):
        super(AffineChannelwiseTransform, self).__init__()
        self.top_half_as_input = top_half_as_input
        self.scale_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.shift_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.net = ResModule(in_dim=in_dim, dim=dim, out_dim=out_dim, res_blocks=res_blocks, bottleneck=bottleneck)

    def forward(self, x):
        if self.top_half_as_input:
            fixed, not_fixed = x.chunk(2, dim=1)
        else:
            not_fixed, fixed = x.chunk(2, dim=1)
        log_scale, shift = self.net(fixed).chunk(2, dim=1)
        log_scale = log_scale.tanh()*self.scale_scale + self.shift_scale

        if self.top_half_as_input:
            x_modified = torch.cat([fixed, not_fixed], dim=1)
            log_scale = torch.cat([log_scale, torch.zeros_like(log_scale)], dim=1)
        else:
            x_modified = torch.cat([not_fixed, fixed], dim=1)
            log_scale = torch.cat([torch.zeros_like(log_scale), log_scale], dim=1)

        return x_modified, log_scale

class ActNorm(nn.Module):
    def __init__(self, n_channels):
        super(ActNorm, self).__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)
        self.shift = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)
        self.n_channels = n_channels
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            self.shift.data = -torch.mean(x, dim=[0,2,3], keepdim=True)
            self.log_scale.data = -torch.log(torch.std(x, [0,2,3], keepdim=True))
            self.initialized = True
            #results = x*torch.exp(self.log_scale) + self.shift
        return x * torch.exp(self.log_scale) + self.shift, self.log_scale


class RealNVP(nn.Module):
    def __init__(self, in_dim, dim, out_dim, res_blocks, bottleneck, size, type):
        super(RealNVP, self).__init__()
        #self.preprocess = Preprocess()
        if type == 'checkerboard':
            self.transforms = nn.ModuleList([AffineCheckerboardTransform(in_dim, dim, out_dim, res_blocks, bottleneck, size, config=1),
                                                       ActNorm(in_dim),
                                                       AffineCheckerboardTransform(in_dim, dim, out_dim, res_blocks, bottleneck, size, config=0),
                                                       ActNorm(in_dim),
                                                       AffineCheckerboardTransform(in_dim, dim, out_dim, res_blocks, bottleneck, size, config=1)])
        elif type == 'channels':
            self.transforms = nn.ModuleList([AffineChannelwiseTransform(in_dim, dim, out_dim, res_blocks, bottleneck, top_half_as_input=True),
                                                     ActNorm(in_dim),
                                                     AffineChannelwiseTransform(in_dim, dim, out_dim, res_blocks, bottleneck, top_half_as_input=True),
                                                     ActNorm(in_dim),
                                                     AffineChannelwiseTransform(in_dim, dim, out_dim, res_blocks, bottleneck, top_half_as_input=True)])
        elif type == 'cycle':
            self.transforms = nn.ModuleList([CouplingLayer(in_dim, dim, out_dim, res_blocks, bottleneck, MaskQuadrant(input_quadrant=0, output_quadrant=1)),
                                             CouplingLayer(in_dim, dim, out_dim, res_blocks, bottleneck, MaskQuadrant(input_quadrant=1, output_quadrant=2)),
                                             CouplingLayer(in_dim, dim, out_dim, res_blocks, bottleneck, MaskQuadrant(input_quadrant=2, output_quadrant=3)),
                                             CouplingLayer(in_dim, dim, out_dim, res_blocks, bottleneck, MaskQuadrant(input_quadrant=3, output_quadrant=0)),
                                             CouplingLayer(in_dim, dim, out_dim, res_blocks, bottleneck,
                                                           MaskQuadrant(input_quadrant=0, output_quadrant=1)),
                                             CouplingLayer(in_dim, dim, out_dim, res_blocks, bottleneck,
                                                           MaskQuadrant(input_quadrant=1, output_quadrant=2)),
                                             CouplingLayer(in_dim, dim, out_dim, res_blocks, bottleneck,
                                                           MaskQuadrant(input_quadrant=2, output_quadrant=3)),
                                             CouplingLayer(in_dim, dim, out_dim, res_blocks, bottleneck,
                                                           MaskQuadrant(input_quadrant=3, output_quadrant=0))
                                             ])
        else:
            print('Error in masking type')
        #self.bn = nn.BatchNorm2d(in_dim)

    def forward(self, z):
        #z = nn.functional.normalize(z)
        z, log_det_J_total = z, torch.zeros(z.size(0)).cuda()
        #z, log_det_J = self.preprocess(z)
        #log_det_J_total += log_det_J
        #self.bn(z)
        for transform in self.transforms:
            z, log_det = transform(z)
            log_det_J_total += log_det

        return z, log_det_J_total

# im = torch.rand(1,256,14,14)
# res = RealNVP(256, 256, 512, 2, True, 16, 'cycle')
# z, log_det = res(im)
# print(z.shape)
# print(log_det.shape)

