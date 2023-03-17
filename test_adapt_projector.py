import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import configuration
import numpy as np
from utils import utils, create_model, prepare_dataset
from models import projector
import copy

def experiment(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    args.corruption = 'gaussian_noise'
    args.adapt = False
    args.adapt_lr = 0.0000001
    args.multi = 5
    args.load = 'cifar10_resnet50RESNETONLY'
    args.adapt = False
    args.niter = 1
    '''--------------------Loading Model-----------------------------'''
    print('Loading model')
    #if args.dataset == 'cifar10':
    #    checkpoint = torch.load(args.root+'weights/BestClassification.pth')
    #    weights = checkpoint['net']
    #elif args.dataset == 'imagenet':
    #    weights = torch.load(args.root + '/weights/' + args.model + '.pth')
    proj_layer = [args.project_layer1, args.project_layer2, args.project_layer3, args.project_layer4]
    model = create_model.create_model(args, proj_layer).to(device)
    checkpoint = torch.load(args.root + 'weights/' + args.load + '.pth') #cifar10_resnet50_all_together.pth')
    ckpt = checkpoint['state_dict']
    # for key in list(ckpt.keys()):
    #     ckpt[key.replace('module.', '')] = ckpt[key]
    #     ckpt.pop(key)
    model.load_state_dict(ckpt) #checkpoint['state_dict']) #, strict=False)
    state = copy.deepcopy(model.state_dict())
    print('Projection:', args.projection)
    print('Projection Layers:', proj_layer)
    print('Number of iterations:', args.niter)

    '''-------------------Optimizer----------------------------------'''
    extractor = projector.get_part(model, args.extract_layer)
    optimizer = torch.optim.Adam(extractor.parameters(), lr=args.adapt_lr)

    '''-------------------Loading Dataset----------------------------'''
    # teloader, _ = prepare_dataset.prepare_test_data(args)

    '''--------------------Test-Time Adaptation----------------------'''
    print('Test-Time Adaptation')
    iteration = [1, 3, 5, 10, 20, 50, 100]
    corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                   'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                   'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'] #, 'cifar_new']
    for level in range(1,5):
        args.level = level
        for corr in corruptions:
            args.corruption = corr
            teloader, _ = prepare_dataset.prepare_test_data(args)
            if args.niter in iteration:
                validation = 3
                indice = iteration.index(args.niter)
                good_good_V = np.zeros([indice + 1, validation])
                good_bad_V = np.zeros([indice + 1, validation])
                bad_good_V = np.zeros([indice + 1, validation])
                bad_bad_V = np.zeros([indice + 1, validation])
                accuracy_V = np.zeros([indice + 1, validation])
                for val in range(validation):
                    good_good = np.zeros([indice + 1, len(teloader.dataset)])
                    good_bad = np.zeros([indice + 1, len(teloader.dataset)])
                    bad_good = np.zeros([indice + 1, len(teloader.dataset)])
                    bad_bad = np.zeros([indice + 1, len(teloader.dataset)])
                    correct = np.zeros(indice + 1)
                    for batch_idx, (inputs, labels) in tqdm(enumerate(teloader)):
                        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                        model.load_state_dict(state)
                        correctness, _ = utils.test_batch(model, inputs, labels, adapt=False)

                        if args.adapt:
                            if args.loss == 1:
                                utils.adapt_batch(model, args.niter, inputs, optimizer, args.K, iteration, args.save_iter, args.projection, proj_layer)
                            if args.loss == 2:
                                utils.adapt_batch_entropy(model, args.niter, inputs, optimizer, args.K, iteration, args.save_iter, args.projection, proj_layer)

                            for k in range(len(iteration[:indice + 1])):
                                ckpt = torch.load(args.save_iter + 'weights_iter_' + str(iteration[k]) + '.pkl')
                                model.load_state_dict(ckpt['weights'])
                                correctness_new, _ = utils.test_batch(model, inputs, labels, adapt=True)
                                for i in range(len(correctness_new.tolist())):
                                    if correctness[i] == True and correctness_new[i] == True:
                                        good_good[k, i + batch_idx * args.batch_size] = 1
                                    elif correctness[i] == True and correctness_new[i] == False:
                                        good_bad[k, i + batch_idx * args.batch_size] = 1
                                    elif correctness[i] == False and correctness_new[i] == True:
                                        bad_good[k, i + batch_idx * args.batch_size] = 1
                                    elif correctness[i] == False and correctness_new[i] == False:
                                        bad_bad[k, i + batch_idx * args.batch_size] = 1
                        else:
                            correct += correctness.sum().item()

                    for k in range(len(iteration[:indice + 1])):
                        correct[k] += np.sum(good_good[k,]) + np.sum(bad_good[k,])
                        accuracy = correct[k] / len(teloader.dataset)
                        good_good_V[k, val] = np.sum(good_good[k,])
                        good_bad_V[k, val] = np.sum(good_bad[k,])
                        bad_good_V[k, val] = np.sum(bad_good[k,])
                        bad_bad_V[k, val] = np.sum(bad_bad[k,])
                        accuracy_V[k, val] = accuracy

                print('Projection:', args.projection)
                print('Projection Layers:', proj_layer)
                print('Extract Layers:', str(args.extract_layer))
                with open('Results/ResNet.txt', 'a') as f:
                    f.write('Projection: ' + str(args.projection) + '\n')
                    f.write('Projection Layers:' + str(proj_layer) + '\n')
                    f.write('Extract Layers:' + str(args.extract_layer) + '\n')
                    f.close()

                for k in range(len(iteration[:indice + 1])):
                    print('--------------------RESULTS----------------------')
                    print('Perturbation: ', args.corruption)
                    print('Nombre d iterations: ', iteration[k])
                    print('Good first, good after: ', str(good_good_V[k,].mean()) + '+/-' + str(good_good_V[k,].std()))
                    print('Good first, bad after: ', str(good_bad_V[k,].mean()) + '+/-' + str(good_bad_V[k,].std()))
                    print('Bad first, good after: ', str(bad_good_V[k,].mean()) + '+/-' + str(bad_good_V[k,].std()))
                    print('Bad first, bad after: ', str(bad_bad_V[k,].mean()) + '+/-' + str(bad_bad_V[k,].std()))
                    print('Accuracy: ', str(accuracy_V[k,].mean()) + '+/-' + str(accuracy_V[k,].std()))

                    with open('Results/ResNet.txt', 'a') as f:
                        f.write('Level: ' + str(args.level) + '\n')
                        f.write('Perturbation: ' + args.corruption + '\n')
                        f.write('Iteration: ' + str(iteration[k]) + '\n')
                        f.write(
                            'Good first, good after: ' + str(good_good_V[k,].mean()) + '+/-' + str(good_good_V[k,].std()) + '\n')
                        f.write('Good first, bad after: ' + str(good_bad_V[k,].mean()) + '+/-' + str(good_bad_V[k,].std()) + '\n')
                        f.write('Bad first, good after: ' + str(bad_good_V[k,].mean()) + '+/-' + str(bad_good_V[k,].std()) + '\n')
                        f.write('Bad first, bad after: ' + str(bad_bad_V[k,].mean()) + '+/-' + str(bad_bad_V[k,].std()) + '\n')
                        f.write('Accuracy: ' + str(accuracy_V[k,].mean()) + '+/-' + str(accuracy_V[k,].std()) + '\n')
                        f.close()

            else:
                good_good = []
                good_bad = []
                bad_good = []
                bad_bad = []
                correct = 0
                for batch_idx, (inputs, labels) in tqdm(enumerate(teloader)):
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    model.load_state_dict(state)
                    correctness, _ = utils.test_batch(model, inputs, labels)

                    if args.adapt:
                        if args.loss == 1:
                            utils.adapt_batch(model, args.niter, inputs, optimizer, args.K, iteration, args.save_iter, args.projection, proj_layer)
                        if args.loss == 2:
                            utils.adapt_batch_entropy(model, args.niter, inputs, optimizer, args.K, iteration, args.save_iter,
                                                      args.projection, proj_layer)
                        correctness_new, _ = utils.test_batch(model, inputs, labels, adapt=True)
                        for i in range(len(correctness_new.tolist())):
                            if correctness[i] == True and correctness_new[i] == True:
                                good_good.append(1)
                            elif correctness[i] == True and correctness_new[i] == False:
                                good_bad.append(1)
                            elif correctness[i] == False and correctness_new[i] == True:
                                bad_good.append(1)
                            elif correctness[i] == False and correctness_new[i] == False:
                                bad_bad.append(1)
                    else:
                        correct += correctness.sum().item()

                correct += len(good_good) + len(bad_good)
                accuracy = correct/len(teloader.dataset)
                print('--------------------RESULTS----------------------')
                print('Perturbation: ', args.corruption)
                print('Accuracy: ', accuracy)
                if args.adapt:
                    print('Good first, good after: ', len(good_good))
                    print('Good first, bad after: ', len(good_bad))
                    print('Bad first, good after: ', len(bad_good))
                    print('Bad first, bad after: ', len(bad_bad))

if __name__ == '__main__':
    args = configuration.argparser()
    experiment(args)
