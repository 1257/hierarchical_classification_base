""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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

def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

def change_labels_to_coarse(dataset, is_new_labels): #для старого набора классов, где порядок был по алфавиту, передать вторым параметром False, для нового - True
    print("is_new_labels in change to coarse is", is_new_labels)
    newset=list(dataset)
    for i in range(len(newset)):
        if i<20:
          print("class", newset[i][1], end=" ")
        buf=list(newset[i])
        
        if(is_new_labels):
          buf[1]=buf[1]//5
        else:
          buf[1]=superclass[buf[1]]
          
        newset[i]=tuple(buf)
        if i<20:
          print("changed to superclass", newset[i][1])
        
    print("superclasses of first 20 items:")
    for i in range(20):
      print(newset[i][1])
    
    print()
    print()
    return newset

def change_labels_order(dataset):
  ll=[]
  for c in range(20):
    l=[i  for i in range(100) if superclass[i]==c]
    ll.append(l)

  mm=[]
  for i in range(20):
    for j in range(5):
      mm.append(ll[i][j])
      
  print("reorder labels")
  newset = list(dataset)
  for i in range(len(newset)):
    if i<20:
      print("class", newset[i][1], end=" ")
    buf=list(newset[i])
    buf[1]=mm.index(buf[1])
    newset[i]=tuple(buf)
    if i<20:
      print("changed to class", newset[i][1])
    
  return newset
  
def get_training_dataloader(is_new_set, mean, std, batch_size=16, num_workers=2, shuffle=True): #True в первом параметре, если порядок классов изменен
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_trainset1, cifar100_trainset2 = torch.utils.data.random_split(cifar100_training, [10000, 40000], generator=torch.Generator().manual_seed(0))
    print('First dataset size:', len(cifar100_trainset2))
    print('Second dataset size:', len(cifar100_trainset1))
    
    
    #if is_new_set:
      #print("using new set in trainloader")
      #cifar100_trainset1 = change_labels_order(cifar100_trainset1)
      #cifar100_trainset2 = change_labels_order(cifar100_trainset2)
    #else:
      #print("using old set in trainloader")
    
    #cifar100_trainset2 = change_labels_to_coarse(cifar100_trainset2, is_new_set)
    
    #cifar100_training_loader1 = DataLoader(
          #cifar100_trainset1, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    #cifar100_training_loader2 = DataLoader(
          #cifar100_trainset2, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    
    #cifar100_trainset2=list(cifar100_trainset2)
    cifar100_trainset2_1=change_labels_to_coarse(cifar100_trainset2, False)
    #cifar100_trainset2_1=change_labels_to_coarse(cifar100_trainset1, False) #only for transfer with 10k+10k
    
    #trainset1_new = change_labels_order(cifar100_trainset1)
    #trainset2_new = change_labels_order(cifar100_trainset2)
    
    #trainset2_super_new=change_labels_to_coarse(trainset2_new, True)
    
    if is_new_set == False:
      cifar100_training_loader1 = DataLoader(
          cifar100_trainset1, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
      cifar100_training_loader2 = DataLoader(
          cifar100_trainset2_1, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
      cifar100_training_loader = DataLoader(
          cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    else:
        cifar100_training_loader1 = DataLoader(
          trainset1_new, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        cifar100_training_loader2 = DataLoader(
          trainset2_super_new, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
        cifar100_training_loader = DataLoader(
          cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader2, cifar100_training_loader1
  
def get_training_dataloader_with_hierarhy(is_new_set, mean, std, batch_size=16, num_workers=2, shuffle=True): 
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_trainset1, cifar100_trainset2 = torch.utils.data.random_split(cifar100_training, [10000, 40000], generator=torch.Generator().manual_seed(0))
    print('First dataset size:', len(cifar100_trainset2))
    print('Second dataset size:', len(cifar100_trainset1))
    
    cifar100_trainset1=list(cifar100_trainset1)
    
    print("two labels examples: class -> superclass")
    for i in range(len(cifar100_trainset1)):
      cifar100_trainset1[i]=list(cifar100_trainset1[i])
      cifar100_trainset1[i].append(superclass[cifar100_trainset1[i][1]])
      cifar100_trainset1[i]=tuple(cifar100_trainset1[i])
      if i<21:
        print(cifar100_trainset1[i][1:3])
        
    #cifar100_trainset1=torch.tensor(cifar100_trainset1)
        
    cifar100_global=torch.cat((torch.tensor(cifar100_trainset1), cifar100_trainset2), 1)
    print("global cifar 100 len:", len(cifar100_global))
        
    #cifar100_trainset2=list(cifar100_trainset2)
    cifar100_trainset2_1=change_labels_to_coarse(cifar100_trainset2, False)
    #cifar100_trainset2_1=change_labels_to_coarse(cifar100_trainset1, False) #only for transfer with 10k+10k
    
    
    cifar100_training_loader1 = DataLoader(
        cifar100_trainset1, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    cifar100_training_loader2 = DataLoader(
        cifar100_trainset2_1, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader2, cifar100_training_loader1

def get_test_dataloader(is_new_set, mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    if is_new_set:
      print("using new set in testloader")
      cifar100_test=change_labels_order(cifar100_test)
    else:
      print("using old set in testloader")
    
    cifar100_test_coarse = change_labels_to_coarse(cifar100_test, is_new_set)
    
    cifar100_test_loader1 = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    cifar100_test_loader2 = DataLoader(
        cifar100_test_coarse, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    return cifar100_test_loader2, cifar100_test_loader1
    
    #cifar100_test_new=change_labels_order(cifar100_test)
    #cifar100_test_super=change_labels_to_coarse(cifar100_test, False)
    #cifar100_test_super_new=change_labels_to_coarse(cifar100_test_new, True)
    
    #for old (alphabetal) set:
    #cifar100_test_loader1 = DataLoader(
        #cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    #cifar100_test_loader2 = DataLoader(
        #cifar100_test_super, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    #for new (grouped) set:
    #cifar100_test_loader1_new = DataLoader(
        #cifar100_test_new, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    #cifar100_test_loader2_new = DataLoader(
        #cifar100_test_super_new, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    #if is_new_set==False:
      #return cifar100_test_loader2, cifar100_test_loader1
    #else:
      #return cifar100_test_loader2_new, cifar100_test_loader1_new

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]
