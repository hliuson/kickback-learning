import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import os
import wandb
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets

from optimizer import *
from hebbian_layer import *
from models import *
from utils import *

               
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--learning_rule', type=str, default=None) #end-to-end backprop, softhebb, influencehebb, random
    parser.add_argument('--dataset', type=str, default='mnist') #mnist, cifar10, cifar100, ms_coco
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_type', type=str, default='mlp-1')
    parser.add_argument('--model_size', type=int, default=128)
    parser.add_argument('--model_depth', type=int, default=5)
    parser.add_argument('--supervised', type=str2bool, default=True) #if false, train unsupervised and then fit linear probe
    parser.add_argument('--task', type=str, default='classification') #if dataset is ms coco, then we can do: a number of different tasks
    parser.add_argument("--influencehebb_soft_y", type=str2bool, default=True) #if true, use softmaxed y for influence hebbian learning rule
    parser.add_argument("--influencehebb_soft_z", type=str2bool, default=True) 
    parser.add_argument("--activation", type=str, default='relu') #relu, mish, triangle
    parser.add_argument("--norm", type=str, default=None) #layer norm, batch norm
    parser.add_argument("--adaptive_lr", type=str2bool, default=False) #if true, use adaptive learning rate
    parser.add_argument("--adaptive_lr_power", type=float, default=0.5) #if true, use adaptive learning rate")
    parser.add_argument("--lr_schedule", type=str, default='constant')
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--init", type=str, default="normal")
    parser.add_argument("--R", type=float, default=2.5)
    parser.add_argument("--influence_type", type=str, default='simple')
    parser.add_argument("--dot_uw", type=str2bool, default=False)
    parser.add_argument("--pooling", type=str, default='max')
    parser.add_argument("--temp", type=float, default=1.0)
    
    args = parser.parse_args()
    
    epochs = args.epochs
    lr = args.lr
    learning_rule = args.learning_rule
    assert learning_rule in ['end2end', 'softhebb', 'softmulthebb', 'influencehebb', 'random']
    
    dataset = args.dataset
    assert dataset in ['mnist', 'cifar10', 'cifar100', 'ms_coco']
    
    batch_size = args.batch_size
    
    model_type = args.model_type
    
    width = args.model_size
    depth = args.model_depth
    
    supervised = args.supervised
    task = args.task
    
    assert depth >= 2
    
    influencehebb_soft_y = args.influencehebb_soft_y
    influencehebb_soft_z = args.influencehebb_soft_z
    
    assert args.activation in ['relu', 'mish', 'triangle']
    act = None
    if args.activation == 'relu':
        act = nn.ReLU
    elif args.activation == 'mish':
        act = nn.Mish
    elif args.activation == 'triangle':
        act = Triangle
        
    assert args.norm in [None, 'layer', 'batch']
    norm = None
    if args.norm == 'layer':
        norm = nn.LayerNorm
    if args.norm == 'batch':
        norm = nn.BatchNorm1d
        if model_type == 'cnn-1':
            norm = nn.BatchNorm2d
        
    assert args.init in ['normal', 'positive-uniform', 'xavier']
    init = args.init
    
    assert args.lr_schedule in ['constant', 'exponential']
    schedule = args.lr_schedule
    
    assert args.influence_type in ['simple', 'grad', "one"]
    influence_type = args.influence_type
    
    assert args.pooling in ['max', 'avg']
    if args.pooling == 'max':
        pool = nn.MaxPool2d
    if args.pooling == 'avg':
        pool = nn.AvgPool2d
    
    if args.adaptive_lr:
        if learning_rule not in ['softhebb', 'softmulthebb', 'influencehebb']:
            raise Exception('adaptive learning rate only works with softhebb and influencehebb')
        if schedule != 'constant':
            raise Exception('adaptive learning rate only works with constant learning rate schedule')

    adaptive_lr_power = args.adaptive_lr_power
    
    name = args.name
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['KICKBACK_DEVICE'] = str(device)
    
    print("Using device: ", device)
    
    train = None
    if dataset == 'mnist':
        train = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
        test = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
        
        mlp_in = 784
        mlp_out = 10
        img_dim = 28
        img_chan = 1
        
    elif dataset == 'cifar10':
        train = torch.utils.data.DataLoader(datasets.CIFAR10('data', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
        test = torch.utils.data.DataLoader(datasets.CIFAR10('data', train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
        
        mlp_in = 3072
        mlp_out = 10
        img_dim = 32
        img_chan = 3
    elif dataset == 'cifar100':
        train = torch.utils.data.DataLoader(datasets.CIFAR100('data', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
        test = torch.utils.data.DataLoader(datasets.CIFAR100('data', train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
        
        mlp_in = 3072
        mlp_out = 100       
    elif dataset == 'ms_coco':
        if task == 'classification':
            train = torch.utils.data.DataLoader(datasets.CocoDetection('data', 'train2017', transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
            test = torch.utils.data.DataLoader(datasets.CocoDetection('data', 'val2017', transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
    
    model = None
    if model_type == 'mlp-1':
        model = MLP1(mlp_in, mlp_out, width, depth, activation=act, normlayer=norm, init=init, init_radius=args.R)
    if model_type == 'cnn-1':
        model = CNN1(img_dim, img_chan, width, mlp_out, depth, activation=act, normlayer=norm, init=init, init_radius=args.R, poollayer=pool)
    if model_type == 'skip-mlp':
        model = SkipMLP(mlp_in, mlp_out, width, depth, activation=act, normlayer=norm, init=init, init_radius=args.R)
    
    if model is None:
        raise Exception('Model is None')
    
    if epochs == -1:
        assert supervised # train-til-convergence only works with supervised learning
        
    
    model = model.to(device)  
    
    tags = [dataset, model_type, learning_rule, f"width-{width}", f"depth-{depth}", f"norm-{args.norm}", f"act-{args.activation}", f"supervised-{supervised}"]
    
    postfix = ""
    if args.name != "":
        postfix = f" | {args.name}"
    
    wandb.init(project="influencehebb", config=args, name=f"{dataset} {model_type} {learning_rule} {depth}x{width}{postfix}", tags=tags)
    
    wandb.watch(model)
    
    print(model)
    
    #add tags to wandb

        
    if epochs == -1:
        train_until_convergence = True
        epochs = 1000000
    else:
        train_until_convergence = False

    opt = HebbianOptimizer(model, lr=lr, learning_rule=learning_rule, supervised=supervised,
                           influencehebb_soft_y=influencehebb_soft_y, influencehebb_soft_z=influencehebb_soft_z, adaptive_lr=args.adaptive_lr, adaptive_lr_p=adaptive_lr_power,
                           schedule=schedule, influence_type=influence_type, dot_uw=args.dot_uw, temp=args.temp)
    

    last_loss = float('inf')
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        for i, (x, y) in enumerate(train):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            if supervised:
                loss = F.cross_entropy(y_hat, y)
                if loss.isnan():
                    raise Exception('Loss is nan')
                loss.backward(retain_graph=True)
                wandb.log({"loss": loss.item()})
            opt.step()
        
        test_loss = 0
        test_acc = 0
        if supervised:
            for i, (x, y) in enumerate(test):
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = F.cross_entropy(y_hat, y)
                test_loss += loss.item()
                
                acc = (y_hat.argmax(dim=1) == y).float().mean()
                test_acc += acc.item()
            
            test_loss /= len(test)
            test_acc /= len(test)
            wandb.log({"test_loss": test_loss, "test_acc": test_acc})
        
        if train_until_convergence:
            if last_loss < test_loss + 0.001:
                break
            last_loss = test_loss
            
    if not supervised:
        print('Completed training, now probing')
        model.head.init_params("xavier", 1)
        opt = torch.optim.SGD(model.head.parameters(), lr=lr)
        for epoch in range(epochs):
            for i, (x, y) in enumerate(train):
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = F.cross_entropy(y_hat, y)
                if loss.isnan():
                    raise Exception('Loss is nan')
                loss.backward(retain_graph=True)
                opt.step()
                wandb.log({"probe_loss": loss.item()})
            
            test_loss = 0
            test_acc = 0
            for i, (x, y) in enumerate(test):
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = F.cross_entropy(y_hat, y)
                test_loss += loss.item()
                
                acc = (y_hat.argmax(dim=1) == y).float().mean()
                test_acc += acc.item()
            
            test_loss /= len(test)
            test_acc /= len(test)
            wandb.log({"probe_test_loss": test_loss, "probe_test_acc": test_acc})


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
    