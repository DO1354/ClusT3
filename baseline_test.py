import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import configuration
import numpy as np
from utils import utils, create_model, prepare_dataset
from models import projector, ResNet
import copy

def experiment(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    cudnn.benchmark = True

    '''--------------------Loading Model-----------------------------'''
    print('Loading model')
    model = ResNet.resnet50(10).cuda()
    #checkpoint = torch.load(args.root + 'weights/' + 'cifar10_resnet50_12_together' + '.pth') #cifar10_resnet50_all_together.pth')
    checkpoint = torch.load('../../Summer2022/TTTFlow/ttt/ttt_together_350_4CLASSES.pth')
    ckpt = checkpoint['net']
    #for key in list(ckpt.keys()):
    #    ckpt[key.replace('module.', '')] = ckpt[key]
    #    ckpt.pop(key)
    model.load_state_dict(ckpt, strict=False) #, strict=False)
    state = copy.deepcopy(model.state_dict())


    '''--------------------Test-Time Adaptation----------------------'''
    print('Test-Time Adaptation')
    common_corruptions = ['original', 'cifar_new', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                          'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                          'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    for corruption in common_corruptions:
        args.corruption = corruption
        teloader, _ = prepare_dataset.prepare_test_data(args)
        correct = 0
        for batch_idx, (inputs, labels) in tqdm(enumerate(teloader)):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            model.load_state_dict(state)
            correctness, _ = utils.test_batch(model, inputs, labels)
            correct += correctness.sum().item()

        accuracy = correct/len(teloader.dataset)
        print('--------------------RESULTS----------------------')
        print('Perturbation: ', args.corruption)
        print('Accuracy: ', accuracy)

if __name__ == '__main__':
    args = configuration.argparser()
    experiment(args)
