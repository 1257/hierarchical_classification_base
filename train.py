# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""
import wandb
import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_single_training_dataloader, get_training_dataloader_with_hierarhy, get_test_dataloader, get_test_dataloader_with_hierarhy, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

#from entropy_2_levels import entropy2lvl
import entropy_2_levels as myEntropy
from models.resnet import ResNet, BasicBlock

superclass = [ 4,  1, 14,  8,  0,  #номер суперкласса соответствует номеру в иерархии на сайте (морские млекопитающие=0, рыбы=1 и т.д.)
               6,  7,  7, 18,  3,  #номер класса соответствует лейблам в датасете
               3, 14,  9, 18,  7, 
              11,  3,  9,  7, 11,  
               6, 11,  5, 10,  7,  
               6, 13, 15,  3, 15,
               0, 11,  1, 10, 12, 
              14, 16,  9, 11,  5,
               5, 19,  8,  8, 15, 
              13, 14, 17, 18, 10,
              16,  4, 17,  4,  2,  
               0, 17,  4, 18, 17,
              10,  3,  2, 12, 12, 
              16, 12,  1,  9, 19,
               2, 10,  0,  1, 16, 
              12,  9, 13, 15, 13,
              16, 19,  2,  4,  6, 
              19,  5,  5,  8, 19,
              18,  1,  2, 15,  6,  
               0, 17,  8, 14, 13]


def train(cifar100_training_loader, warmup_scheduler, epoch, loss_function, optimizer, useClasses, useSuperclasses):
    start = time.time()
    net.train()
    for batch_index, (images, labels, class_labels) in enumerate(cifar100_training_loader):  #class_labels is optional, for two-level
        labs=labels
        
        if args.gpu:        
            labels = labels.cuda()
            class_labels = class_labels.cuda()  #two-level
            images = images.cuda()
        
        optimizer.zero_grad()
        outputs = net(images)

        #print("outputs size:", outputs.size())
        #print("outputsSuper size:", outputsSuper.size())
        #print("labels", labels)
        #print("class_labels", class_labels)
        
        loss = loss_function(outputs, labels, class_labels, useSuperclasses, useClasses)

        wandb.log({"loss": loss})
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(loss_function, cifar100_test_loader, epoch=0, tb=True, ):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct1 = 0.0
    correct2 = 0.0

    for (images, labels, class_labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
            class_labels = class_labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels, class_labels, True, True)

        test_loss += loss.item()
        _, preds = outputs.max(1)
                
        preds_super = [superclass[preds[i]] for i in range(len(preds)) if class_labels[i]!=-1]
        class_labels = [class_labels[i] for i in range(len(class_labels)) if class_labels[i]!=-1]
        
        class_labels = torch.tensor(class_labels)
        class_labels = class_labels.cuda()
        
        preds_super = torch.tensor(preds_super)
        preds_super = preds_super.cuda()
        
        correct1 += preds.eq(torch.tensor(class_labels)).sum()
        correct2 += preds_super.eq(torch.tensor(labels)).sum()
        
        #print("real classes:", class_labels)
        #print("preds:", preds)
        
        #print("real superclasses:", labels)
        #print("preds_super:", preds_super)
        
    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy100: {:.4f}, Accuracy20: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        #correct1.float() / len(cifar100_test_loader.dataset),
        correct1.float() / 10000.0,
        correct2.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        #writer.add_scalar('Test/Accuracy100', correct1.float() / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy100', correct1.float() / 10000.0, epoch)
        writer.add_scalar('Test/Accuracy20', correct2.float() / len(cifar100_test_loader.dataset), epoch)

    #return correct1.float() / len(cifar100_test_loader.dataset), correct2.float() / len(cifar100_test_loader.dataset)
    return correct1.float() / 10000.0, correct2.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    wandb.init(project="two_steps", entity="hierarchical_classification")
    wandb.config = {"epochs": 200, "batch_size": 128}
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = get_network(args)
    if torch.cuda.is_available():
        net=ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100).cuda()
    #net.set_output_size(20)
    
    wandb.log({"experiment": settings.EXPERIMENT})
    wandb.log({"fine set size": settings.COMPLEX_TRAINSET_SIZE})
    
    #data preprocessing:
    if settings.EXPERIMENT == "baseline":
        cifar100_training_loader2 = get_single_training_dataloader(
            False,
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            num_workers=4,
            batch_size=args.b,
            shuffle=True            
        )
        
    elif settings.EXPERIMENT == "finetune ":
        cifar100_training_loader1, cifar100_training_loader2 = get_training_dataloader( #loader1 - superclasses, loader2 - classes
        settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            num_workers=4,
            batch_size=args.b,
            shuffle=True
        )
        
    elif settings.EXPERIMENT == "hierarchy":
        cifar100_training_loader2 = get_training_dataloader_with_hierarhy(
            False,
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            num_workers=4,
            batch_size=args.b,
            shuffle=True
        )
    
    #get testloader with classes and superclasses (to get accuracy at 20 and 100 classes)
    cifar100_test_loader = get_test_dataloader_with_hierarhy(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    loss_function1 = nn.CrossEntropyLoss()
    loss_function2 = myEntropy.entropy2lvl
    
    if settings.EXPERIMENT == "finetune ":
        optimizer1 = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        train_scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
        iter_per_epoch1 = len(cifar100_training_loader1)
        warmup_scheduler1 = WarmUpLR(optimizer1, iter_per_epoch1 * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    wandb.log({"stage": 1})
    # step 1
    #for epoch in range(1, settings.EPOCH + 1):
    if settings.EXPERIMENT == "finetune ":
        for epoch in range(1, 51):  #51
            if epoch > args.warm:
                train_scheduler1.step(epoch)

            if args.resume:
                if epoch <= resume_epoch:
                    continue

            train(cifar100_training_loader1, warmup_scheduler1, epoch, loss_function1, optimizer1, useSuperclasses=True, useClasses=False)
            acc100, acc20 = eval_training(loss_function1, cifar100_test_loader, epoch)
            wandb.log({"accuracy 100": acc100})
            wandb.log({"accuracy 20": acc20})
            wandb.log({"stage": 1})

            #start to save best performance model after learning rate decay to 0.01
            if epoch > settings.MILESTONES[1] and best_acc < acc100:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)
                best_acc = acc100
                continue

            if not epoch % settings.SAVE_EPOCH:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)

    #net.set_output_size(100)
    #net.freeze()
    net=net.cuda()
    iter_per_epoch2 = len(cifar100_training_loader2)
    optimizer2 = optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, milestones=settings.MILESTONES1, gamma=0.2) #learning rate decay
    warmup_scheduler2 = WarmUpLR(optimizer2, iter_per_epoch2 * args.warm)
    #print(filter(lambda x: x.requires_grad, net.parameters()))
    
    wandb.log({"stage": 2})
    # step 2
    #for epoch in range(1, settings.EPOCH + 1):
    if settings.EXPERIMENT == "finetune ":
        for epoch in range(1, 21):  #21
            if epoch > args.warm:
                train_scheduler2.step(epoch)

            if args.resume:
                if epoch <= resume_epoch:
                    continue

            train(cifar100_training_loader2, warmup_scheduler2, epoch, loss_function2, optimizer2, useSuperclasses=False, useClasses=True)  #loss_function2
            acc100, acc20 = eval_training(loss_function2, cifar100_test_loader, epoch) #loss_function2
            wandb.log({"accuracy 100": acc100})
            wandb.log({"accuracy 20": acc20})
            wandb.log({"stage": 2})

            #start to save best performance model after learning rate decay to 0.01
            if epoch > settings.MILESTONES1[1] and best_acc < acc100:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)
                best_acc = acc100
                continue

            if not epoch % settings.SAVE_EPOCH:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)
            
        
    
    net.conv5_x.requires_grad_(True)
    net.conv4_x.requires_grad_(True)
    net.conv3_x.requires_grad_(True)
    net.conv2_x.requires_grad_(True)
    net.conv1.requires_grad_(True)
    net.avg_pool.requires_grad_(True)
    net.fc.requires_grad_(True)
    
    optimizer2 = optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, milestones=settings.SMALL_MILESTONES, gamma=0.2) #learning rate decay, MILESTONES
    warmup_scheduler2 = WarmUpLR(optimizer2, iter_per_epoch2 * args.warm)
    #print(filter(lambda x: x.requires_grad, net.parameters()))
    
    print(net)
    
    wandb.log({"stage": 3})
    # step 3
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler2.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(cifar100_training_loader2, warmup_scheduler2, epoch, loss_function2, optimizer2, False, True) #loss_function2
        acc100, acc20 = eval_training(loss_function2, cifar100_test_loader, epoch) #loss_function2
        wandb.log({"accuracy 100": acc100})
        wandb.log({"accuracy 20": acc20})
        wandb.log({"stage": 3})

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.SMALL_MILESTONES [1] and best_acc < acc100:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc100
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
    
    writer.close()
