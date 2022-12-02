import timm
import torch
import torch.nn.functional as F
from types import MethodType
from models import adapters, ResNet, ResNet_projector

def model_sizes(args, layer):
    if args.dataset == 'imagenet':
        if layer == 0:
            channels, resolution = 64, 112
        if layer == 1:
            channels, resolution = 256, 56
        if layer == 2:
            channels, resolution = 512, 28
        if layer == 3:
            channels, resolution = 1024, 14
        if layer == 4:
            channels, resolution = 2048, 7

    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if layer == 0:
            channels, resolution = 64, 32
        if layer == 1:
            channels, resolution = 256, 32
        if layer == 2:
            channels, resolution = 512, 16
        if layer == 3:
            channels, resolution = 1024, 8
        if layer == 4:
            channels, resolution = 2048, 4

    return channels, resolution

#This is the modified forward_features method from the timm model (special case for CIFAR-10/100)
#Ignore the error, as the function checkpoint_seq is out of context, but is correct inside timm model
def cifar_forward_features(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    if self.grad_checkpointing and not torch.jit.is_scripting():
        x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
    else:
        x = self.layer1(x)
        if hasattr(self, 'mask1'):
            x = self.mask1(x, inference=True if self.inference else False)
        x = self.layer2(x)
        if hasattr(self, 'mask2'):
            x = self.mask2(x, inference=True if self.inference else False)
        x = self.layer3(x)
        if hasattr(self, 'mask3'):
            x = self.mask3(x, inference=True if self.inference else False)
        x = self.layer4(x)
        if hasattr(self, 'mask4'):
            x = self.mask4(x, inference=True if self.inference else False)
    return x

def forward_head(self, x, pre_logits: bool = False):
    x = self.global_pool(x)
    print(x.shape)
    if self.drop_rate:
        x = torch.nn.functional.dropout(x, p=float(self.drop_rate), training=self.training)
    return x if pre_logits else self.fc(x)

#This is the modified forward_features method from the timm model (special case for ImageNet)
#Ignore the error, as the function checkpoint_seq is out of context, but is correct inside timm model
def imagenet_forward_features(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)

    if self.grad_checkpointing and not torch.jit.is_scripting():
        x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
    else:
        x = self.layer1(x)
        if hasattr(self, 'mask1'):
            x = self.mask1(x, inference=True if self.inference else False)
        x = self.layer2(x)
        if hasattr(self, 'mask2'):
            x = self.mask2(x, inference=True if self.inference else False)
        x = self.layer3(x)
        if hasattr(self, 'mask3'):
            x = self.mask3(x, inference=True if self.inference else False)
        x = self.layer4(x)
        if hasattr(self, 'mask4'):
            x = self.mask4(x, inference=True if self.inference else False)
    return x

def cifar_forward(self, x, adapt=False, feature=False):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    if hasattr(self, 'mask1') and adapt:
        out = self.mask1(out, inference=True if self.inference else False)
    out = self.layer2(out)
    if hasattr(self, 'mask2') and adapt:
        out = self.mask2(out, inference=True if self.inference else False)
    out = self.layer3(out)
    if hasattr(self, 'mask3') and adapt:
        out = self.mask3(out, inference=True if self.inference else False)
    out = self.layer4(out)
    if hasattr(self, 'mask4') and adapt:
        out = self.mask4(out, inference=True if self.inference else False)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    features = out
    out = self.fc(out)
    if feature:
        return out, features
    else:
        return out

def get_mask_adapter(name, channels, resolution):
    '''
    :param name: type of adapter ('mask' for binary only, 'maskadapt' for binary + adapter)
    :param size: resolution of mask/adapter
    :return:
    '''
    if name == 'mask':
        adapter = adapters.MaskAttention(channels, resolution)
    elif name == 'maskadapt':
        adapter = adapters.MaskAdapter(channels, resolution)
    elif name == 'conv':
        adapter = adapters.ConvAdapter(channels)
    elif name == 'flow':
        adapter = adapters.FlowAdapter(channels, resolution)
    return adapter

def create_model(args, layers, proj_layers, weights=None):
    '''
    :param dataset: dataset to use (CIFAR-10, ImageNet)
    :param layers: where to put mask/adapters. Each element with the form (TYPE, SIZE)
    :return: timm model + MaskUp
    '''
    #Creating model based on dataset
    if args.dataset == 'cifar10':
        num_classes = 10
        if args.timm:
            model = timm.create_model(args.model, num_classes=num_classes, pretrained=False)
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        else:
            model = ResNet_projector.resnet50(num_classes, layers, proj_layers)
    elif args.dataset == 'cifar100':
        model = timm.create_model(args.model, num_classes=100, pretrained=False)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    elif args.dataset == 'imagenet':
        model = timm.create_model('resnet50', num_classes=1000, pretrained=False)
    model.inference=True

    #Loading weights
    if weights is not None:
        model.load_state_dict(weights, strict=False)

    #Adding masks/adapters
    '''if layers[0] is not None:
        channels, resolution = model_sizes(args, 1)
        model.mask1 = get_mask_adapter(name=layers[0], channels=channels, resolution=resolution)
    if layers[1] is not None:
        channels, resolution = model_sizes(args, 2)
        model.mask2 = get_mask_adapter(name=layers[1], channels=channels, resolution=resolution)
    if layers[2] is not None:
        channels, resolution = model_sizes(args, 3)
        model.mask3 = get_mask_adapter(name=layers[2], channels=channels, resolution=resolution)
    if layers[3] is not None:
        channels, resolution = model_sizes(args, 4)
        model.mask4 = get_mask_adapter(name=layers[3], channels=channels, resolution=resolution)'''

    #Changing forward function in timm model (only if masking is specified)
    '''if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if args.timm:
            model.forward_features = MethodType(cifar_forward_features, model)
        else:
            model.forward = MethodType(cifar_forward, model)
    elif args.dataset == 'imagenet':
        model.forward_features = MethodType(imagenet_forward_features, model)'''

    return model

def load_state_dict(model, state_dict):
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in model_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        model_state[name].copy_(param)
    print(model_state.keys())
