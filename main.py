import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import os
import wandb
import sys
import argparse
import uuid
import json
from tqdm import tqdm

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

def run(**kwargs):
    
        
    epochs = kwargs.get('epochs', 10)
    lr = kwargs.get('lr', 0.1)
    learning_rule = kwargs.get('learning_rule', 'simplesofthebb')
    assert learning_rule in ['end2end', 'softhebb', 'influencehebb', 'random', 'simplesofthebb']
    
    if learning_rule == 'random':
        epochs = 0
    
    dataset = kwargs.get('dataset', 'mnist')
    assert dataset in ['mnist', 'cifar10', 'cifar100', 'ms_coco']
    
    batch_size = kwargs.get('batch_size', 32)
    
    model_type = kwargs.get('model_type', 'mlp-1')
    
    width = kwargs.get('width', 32)
    depth = kwargs.get('depth', 2)
    
    supervised = kwargs.get('supervised', True)
    task = kwargs.get('task', 'classification')
    
    assert depth >= 2
    
    influencehebb_soft_y = kwargs.get('influencehebb_soft_y', True)
    influencehebb_soft_z = kwargs.get('influencehebb_soft_z', True)
    
    activation = kwargs.get('activation', 'relu')
    
    assert activation in ['relu', 'mish', 'triangle']
    act = None
    if activation == 'relu':
        act = nn.ReLU
    elif activation == 'mish':
        act = nn.Mish
    elif activation == 'triangle':
        act = Triangle
    
    norm = kwargs.get('norm', None)
    assert norm in [None, 'layer', 'batch']
    norm = None
    if norm == 'layer':
        norm = nn.LayerNorm
        if model_type == 'cnn-1':
            norm = nn.InstanceNorm2d
    if norm == 'batch':
        norm = nn.BatchNorm1d
        if model_type == 'cnn-1':
            norm = nn.BatchNorm2d
        
        
    init = kwargs.get('init', 'normal')
    assert init in ['normal', 'positive-uniform', 'xavier']
    
    lr_schedule = kwargs.get('lr_schedule', 'constant')
    assert lr_schedule in ['constant', 'exponential']

    influence_type = kwargs.get('influence_type', 'grad')
    assert args.influence_type in ['simple', 'grad', "one", "random", "full_random"]
    
    pooling = kwargs.get('pooling', 'max')
    assert pooling in ['max', 'avg']
    if pooling == 'max':
        pool = nn.MaxPool2d
    if pooling == 'avg':
        pool = nn.AvgPool2d

    adaptive_lr_power = kwargs.get('adaptive_lr_power', 0.5)
    
    name = kwargs.get('name', '')
    R = kwargs.get('R', 5)
    dot_uw = kwargs.get('dot_uw', False)
    n_trials = kwargs.get('n_trials', 1)
    save = kwargs.get('save', False)
    const_feedback = kwargs.get('const_feedback', False)
    dropout = kwargs.get('dropout', False)
    probe = kwargs.get('probe', False)
    exp_avg = kwargs.get('exp_avg', 0.95)
    group_size = kwargs.get('group_size', -1)
    shuffle = kwargs.get('shuffle', False)
    temp = kwargs.get('temp', 1.0)
    adaptive_lr = kwargs.get('adaptive_lr', False)
    
    
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
        model = MLP1(mlp_in, mlp_out, width, depth, activation=act, normlayer=norm, init=init, init_radius=R, dropout=dropout)
    if model_type == 'cnn-1':
        model = CNN1(img_dim, img_chan, width, mlp_out, depth, activation=act, normlayer=norm, init=init, init_radius=R, poollayer=pool)
    if model_type == 'skip-mlp':
        model = SkipMLP(mlp_in, mlp_out, width, depth, activation=act, normlayer=norm, init=init, init_radius=R)
    
    if model is None:
        raise Exception('Model is None')
    
    if epochs == -1:
        assert supervised # train-til-convergence only works with supervised learning
        
    
    model = model.to(device)  
    print(model)
    
        
    if epochs == -1:
        train_until_convergence = True
        epochs = 1000000
    else:
        train_until_convergence = False

    opt = HebbianOptimizer(model, lr=lr, learning_rule=learning_rule, supervised=supervised,
                           influencehebb_soft_y=influencehebb_soft_y, influencehebb_soft_z=influencehebb_soft_z, adaptive_lr=adaptive_lr, adaptive_lr_p=adaptive_lr_power,
                           schedule=lr_schedule, influence_type=influence_type, dot_uw=dot_uw, temp=temp, const_feedback=const_feedback,
                           exp_avg=exp_avg, group_size=group_size, shuffle=shuffle)
    

    last_loss = float('inf')
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        model.train()
        for i, (x, y) in tqdm(enumerate(train)):
            if i % 100 == 0:
                show_neurons(model, model_type, dataset)
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
            model.eval()
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
            print(f"Test loss: {test_loss}, Test acc: {test_acc}")
        
        opt.inc_epoch()
        if train_until_convergence:
            if last_loss < test_loss + 0.001:
                break
            last_loss = test_loss
    
    if probe:
        print('Completed training, now probing')
        model.head.init_params("xavier", 1)
        opt = torch.optim.AdamW(model.head.parameters())
        for epoch in tqdm(range(3)):
            model.train()
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
            model.eval()
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
            print(f"Probe test loss: {test_loss}, Probe test acc: {test_acc}")
       
    
    if save:
        uid = uuid.uuid4()
        torch.save(model.state_dict(), f"models/{wandb.run.name}_{uid}.pt")
        #save args as JSON
        #with open(f"models/{wandb.run.name}_{uid}.json", 'w') as f:
            #json.dump(vars(kw), f)
    
    # Select the filter index of the neuron you want to visualize
    #filter_index = 0

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)

    # Visualize the receptive field
    #layer = model.layers[-1]
    #input_image = visualize_receptive_field(model, layer, filter_index, device)

    # Plot the image
    #plt.imshow(input_image)
    #plt.axis('off')
    #plt.show()

def show_neurons(model, model_type, dataset):
    if model_type == 'mlp-1': #visualize weights
        weights = model.layers[0].weight.detach().cpu().numpy()
        neurons = weights[:25]
        if dataset == 'mnist':
            neurons = neurons.reshape(25, 28, 28)
        if dataset == 'cifar10':
            neurons = neurons.reshape(25, 32, 32, 3)
            #normalize to 0-1 separately for each neuron
            neurons = (neurons - neurons.min(axis=(1, 2, 3), keepdims=True)) / (neurons.max(axis=(1, 2, 3), keepdims=True) - neurons.min(axis=(1, 2, 3), keepdims=True))
    if model_type == 'cnn-1': #visualize 25 filters
        neurons = model.layers[0].filter.detach().cpu().numpy() #by default cnn filters are (out, in, h, w)
        neurons = neurons[:25]
        if dataset == 'mnist':
            neurons = neurons.transpose(0, 2, 3, 1) #we want (out, h, w, in)
            neurons = neurons.reshape(25, 3, 3)
        if dataset == 'cifar10':
            # we want (out, h, w, in)
            neurons = neurons.transpose(0, 2, 3, 1) 
            neurons = neurons.reshape(25, 3, 3, 3)
            #normalize to 0-1 separately for each neuron
            neurons = (neurons - neurons.min(axis=(1, 2, 3), keepdims=True)) / (neurons.max(axis=(1, 2, 3), keepdims=True) - neurons.min(axis=(1, 2, 3), keepdims=True)) 
        
    fig, ax = plt.subplots(5, 5)
    for i in range(5):
        for j in range(5):
            ax[i, j].imshow(neurons[i*5 + j])
            ax[i, j].axis('off')
    wandb.log({"neurons": fig})
    plt.close(fig)
    
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def visualize_receptive_field(model, layer, filter_index, device, num_iterations=100, learning_rate=1e-1, tv_weight=1e-3):
    model.eval()

    input_image = torch.randn(1, 3, 224, 224, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([input_image], lr=learning_rate)

    activation = None
    def store_activation(module, input, output):
        nonlocal activation
        activation = output

    hook_handle = layer.register_forward_hook(store_activation)

    for i in range(num_iterations):
        optimizer.zero_grad()
        model(input_image)
        neuron_activation = activation[:, filter_index]
        loss = -torch.mean(neuron_activation)

        # Add Total Variation regularization
        tv_reg = tv_weight * (torch.sum(torch.abs(input_image[:, :, 1:, :] - input_image[:, :, :-1, :])) +
                              torch.sum(torch.abs(input_image[:, :, :, 1:] - input_image[:, :, :, :-1])))

        loss += tv_reg
        
        loss.backward()
        optimizer.step()

        # Clip the input image to maintain values between [0, 1]
        input_image.data.clamp_(0, 1)

    hook_handle.remove()

    input_image = input_image.detach().squeeze(0).cpu()
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    input_image = transforms.ToPILImage()(input_image)

    return input_image






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
    parser.add_argument("--R", type=float, default=5)
    parser.add_argument("--influence_type", type=str, default='simple')
    parser.add_argument("--dot_uw", type=str2bool, default=False)
    parser.add_argument("--pooling", type=str, default='max')
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--save", type=str2bool, default=False)
    parser.add_argument("--const_feedback", type=str2bool, default=True)
    parser.add_argument("--dropout", type=str2bool, default=False)
    parser.add_argument("--probe", type=str2bool, default=False)
    parser.add_argument("--exp_avg", type=float, default=0.95)
    parser.add_argument("--group_size", type=int, default=-1)
    parser.add_argument("--shuffle", type=str2bool, default=False)
    
    args = parser.parse_args()
    tags = [args.dataset, args.model_type, args.learning_rule, f"width-{args.model_size}", f"depth-{args.model_depth}", f"norm-{args.norm}", f"act-{args.activation}", f"supervised-{args.supervised}"]
    
    
    for i in range(args.n_trials):
        postfix = ""
        if args.name != "":
            postfix = f" | {args.name}"
        if args.n_trials > 1:
            postfix += f" | trial {i+1}/{args.n_trials}"
        wandb.init(project="influencehebb", config=args, name=f"{args.dataset} {args.model_type} {args.learning_rule} {args.model_depth}x{args.model_size}{postfix}", tags=tags)
        run(**vars(args))


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
    