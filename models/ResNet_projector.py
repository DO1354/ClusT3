import torch
import torch.nn as nn
import torch.nn.functional as F

from models.adapters import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, layers=[None,None,None,None], projector=[None, None, None, None]):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) #64
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) #128
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) #256
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) #512
        self.fc = nn.Linear(512*block.expansion, num_classes) #512
        self.layers = layers
        if layers[0] is not None:
            self.mask1 = self._get_mask_adapter(layers[0], 256, 32)
        if layers[1] is not None:
            self.mask2 = self._get_mask_adapter(layers[1], 512, 16)
        if layers[2] is not None:
            self.mask3 = self._get_mask_adapter(layers[2], 1024, 8)
        if layers[3] is not None:
            self.mask4 = self._get_mask_adapter(layers[3], 2048, 4)
        if projector[0] is not None:
            self.projector1 = nn.Linear(256*32*32, num_classes)
        if projector[1] is not None:
            self.projector2 = nn.Linear(512*16*16, num_classes)
        if projector[2] is not None:
            self.projector3 = nn.Linear(1024*8*8, num_classes)
        if projector[3] is not None:
            self.projector4 = nn.Linear(2048*4*4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _get_mask_adapter(self, name, channels, resolution):
        if name == 'mask':
            adapter = MaskAttention(channels, resolution)
        elif name == 'maskadapt':
            adapter = MaskAdapter(channels, resolution)
        elif name == 'alladapt':
            adapter = AllAdapter(channels, resolution)
        elif name == 'conv':
            adapter = ConvAdapter(channels)
        elif name == 'cycle':
            adapter = CycleCL(channels)
        elif name == 'flow':
            adapter = FlowAdapter(channels, resolution, C=1)
        return adapter

    def forward(self, x, adapt=False, feature=False, projection=False):
        out_proj = [None, None, None, None]
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if hasattr(self, 'mask1') and adapt:
            out = self.mask1(out, inference=True if self.inference else False)
        if hasattr(self, 'projector1'):
            z1 = out.view(out.size(0), -1)
            z1 = self.projector1(z1)
            out_proj[0] = z1
        out = self.layer2(out)
        if hasattr(self, 'mask2') and adapt:
            out = self.mask2(out, inference=True if self.inference else False)
        if hasattr(self, 'projector2'):
            z2 = out.view(out.size(0), -1)
            z2 = self.projector2(z2)
            out_proj[1] = z2
        out = self.layer3(out)
        if hasattr(self, 'mask3') and adapt:
            out = self.mask3(out, inference=True if self.inference else False)
        if hasattr(self, 'projector3'):
            z3 = out.view(out.size(0), -1)
            z3 = self.projector3(z3)
            out_proj[2] = z3
        out = self.layer4(out)
        if hasattr(self, 'mask4') and adapt:
            out = self.mask4(out, inference=True if self.inference else False)
        if hasattr(self, 'projector4'):
            z4 = out.view(out.size(0), -1)
            z4 = self.projector4(z4)
            out_proj[3] = z4
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        features = out
        out = self.fc(out)
        if feature:
            if projection:
                return out, features, out_proj
            else:
                return out, features
        else:
            if projection:
                return out, out_proj
            else:
                return out

def resnet50(num_classes = 10, layers = [None, None, None, None], projector = [None, None, None, None],  **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, layers=layers, projector=projector, **kwargs)
    return model
