import torch
import torch.nn as nn
from models import RealNVP
from models import CycleRealNVP

class Threshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m):
        return (m > 0.5).float()

    @staticmethod
    def backward(ctx, m):
        return m

class CycleCL(nn.Module):
    def __init__(self, channels, C=1):
        super(CycleCL, self).__init__()
        self.flow = CycleRealNVP.RealNVP(channels, channels, channels*2, 1, True, 16, 'cycle')

    def forward(self, x, inference=False):
        return self.flow(x)[0]

'''Using only the binary mask for attention (no adapter multiplied after)'''
class MaskAttention(nn.Module):
    def __init__(self, channels, size):
        super(MaskAttention, self).__init__()
        self.mask = nn.Parameter(torch.rand(channels, size,size), requires_grad=True)
        self.τ = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x, inference=False):
        if not inference:
            u = torch.rand(1)
            l = torch.log(u) - torch.log(1-u)
            p_mask = torch.sigmoid((self.mask + l.to(x.device))/self.τ)
        else:
            p_mask = torch.sigmoid((self.mask)/self.τ)
        b_mask = Threshold.apply(p_mask)

        return x*b_mask

'''Binary mask + adapter. Option "keep=True" enables that unselected features remain the same, otherwise they are zero'''
class MaskAdapter(nn.Module):
    def __init__(self, channels, size):
        super(MaskAdapter, self).__init__()
        self.mask = nn.Parameter(torch.ones(channels, size, size), requires_grad=True)
        self.τ = nn.Parameter(torch.rand(1), requires_grad=True)
        self.adapter = nn.Parameter(torch.ones(channels,size,size), requires_grad=True)

    def forward(self, x, inference=False, keep=True):
        if not inference:
            u = torch.rand(1)
            l = torch.log(u) - torch.log(1-u)
            p_mask = torch.sigmoid((self.mask + l.to(x.device))/self.τ)
        else:
            p_mask = torch.sigmoid((self.mask)/self.τ)
        b_mask = Threshold.apply(p_mask)
        x_masked = x*b_mask
        x_adapt = x_masked*self.adapter
        if keep:
            x_normal = x*(1 - b_mask) #torch.logical_xor(torch.ones(x.shape).to(x.device), b_mask)
            x_adapt = x_adapt + x_normal

        return x_adapt

class AllAdapter(nn.Module):
    def __init__(self, channels, size):
        super(AllAdapter, self).__init__()
        self.adapter = nn.Parameter(torch.ones(channels,size,size), requires_grad=True)

    def forward(self, x, inference=False, keep=True):
        x_adapt = x*self.adapter
        return x_adapt

'''Binary mask + adapter. Option "keep=True" enables that unselected features remain the same, otherwise they are zero'''
class MatrixMaskAdapter(nn.Module):
    def __init__(self, size):
        super(MatrixMaskAdapter, self).__init__()
        self.mask = nn.Parameter(torch.rand(size, size), requires_grad=True)
        self.τ = nn.Parameter(torch.randn(1), requires_grad=True)
        self.adapter = nn.Parameter(torch.rand(1,size,size), requires_grad=True)

    def forward(self, x, inference=False, keep=True):
        if not inference:
            u = torch.rand(1)
            l = torch.log(u) - torch.log(1-u)
            p_mask = torch.sigmoid((self.mask + l.to(x.device))/self.τ)
        else:
            p_mask = torch.sigmoid((self.mask)/self.τ)
        b_mask = Threshold.apply(p_mask)
        x_masked = x*b_mask
        x_adapt = x_masked*self.adapter
        if keep:
            x_normal = x*torch.logical_xor(torch.ones(x.shape).to(x.device), b_mask)
            x_adapt = x_adapt + x_normal

        return x_adapt

class FlowAdapter(nn.Module):
    def __init__(self, channels, size, C=1):
        super(FlowAdapter, self).__init__()
        self.flow = RealNVP.realnvp(channels, size, K=C)
        self.inference = True

    def forward(self,x, inference=True):
        return self.flow(x)[0]

class ConvAdapter(nn.Module):
    def __init__(self, channels):
        super(ConvAdapter, self).__init__()
        self.adapter = nn.Sequential(nn.Conv2d(channels, 2*channels, 1, bias=False),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(2*channels, channels, 1, bias=False))

    def forward(self, x, inference=True):
        return self.adapter(x)

#Play with it!
'''Mask Attention'''
#x = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]).unsqueeze(0)
#x = torch.cat((x,x,x), dim=0).unsqueeze(0)
#mask = MaskAttention(size=3)
#x_mask = mask(x)
#criterion = nn.MSELoss()
#loss = criterion(x, x_mask)
#loss.backward()
#print(loss)


'''Mask Adapter'''
#mask_adapt = MaskAdapter(size=2)
#x_mask = mask_adapt(x)

#print(x_mask)
