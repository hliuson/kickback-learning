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

from abc import ABC, abstractmethod

def kickback_get_signs_for_length(length: int):
        #first half of the signs are positive, second half are negative]
        ones = torch.ones(length)
        #ones[length//2:] = -1
        return ones
    
class PNReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.relu(x) * kickback_get_signs_for_length(x.shape[1]).to(x.device)

class KickbackNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, activation=PNReLU, conv=False, img_size=28):
        super().__init__()
        self.torso = torch.nn.Sequential()
        self.linears = []
        self.image_size = img_size
        self.in_dim = in_dim
        for i in range(num_layers):
            layer = None
            norm = None
            if i == 0:
                norm = torch.nn.LayerNorm(in_dim)
                layer = HebbianLinear(in_dim, hidden_dim)
                if conv:
                    layer = HebbianConv2d(in_dim, hidden_dim, 3, 1, 1)
                    norm = torch.nn.LayerNorm([in_dim, img_size, img_size])
            else:
                norm = torch.nn.LayerNorm(hidden_dim)
                layer = HebbianLinear(hidden_dim, hidden_dim)
                if conv:
                    layer = HebbianConv2d(hidden_dim, hidden_dim, 3, 1, 1)
                    dim = img_size // min(4, 2**i)
                    norm = torch.nn.LayerNorm([hidden_dim, dim, dim])
            self.torso.add_module(f'layernorm_{i}', norm)
            self.torso.add_module(f'layer_{i}', layer)
            self.torso.add_module(f'activation_{i}', activation())
            
            #pool 28x28 -> 14x14 -> 7x7, and then stop pooling
            if conv and i < 2:
                self.torso.add_module(f'pool_{i}', torch.nn.MaxPool2d(2))
            
            self.linears += [layer]
            
        self.imgdim = img_size // min(4, 2**num_layers) #28 -> 14 -> 7
        self.head = torch.nn.Sequential()
        if conv:
            linear = nn.Linear(hidden_dim * self.imgdim * self.imgdim, out_dim)
        else:
            linear = nn.Linear(hidden_dim, out_dim)
        self.head.add_module('linear', linear)
        self.head.add_module('activation', torch.nn.Sigmoid())
        self.conv = conv
        self.linears += [linear]
        
    def forward(self, x, endtoend=False):
        if self.conv:
            x = x.view(x.shape[0], self.in_dim, self.image_size, self.image_size)
        else:
            x= x.view(x.shape[0], self.in_dim)
        if endtoend:
            x = self.torso(x)
        else:
            with torch.no_grad():
                x = self.torso(x)
        x = x.view(x.shape[0], -1)
        x = self.head(x)
        return x
    
    def full_kickback_rule(self, signal, rate=0.1):
        device = os.environ['KICKBACK_DEVICE']
        linear_no = 0
        for i, layer in enumerate(self.torso):
            if isinstance(layer, HebbianLinear):
                #kickback rule is dw_ij = (global error) *  (downstream influence of y_j) * (x_i) * sgn_j
                #sgn_j denotes whether y_j is a positive or negative unit, and also if the neuron fires or not
                x = layer.x
                y = layer.y
                outsigns = kickback_get_signs_for_length(layer.y.shape[1]).to(device)
                sgn = outsigns * (y > 0).float().to(device)
                #influence of a neuron j is the sum over wJk * sgn_k
                
                next_layer = self.linears[linear_no+1]                
                
                influence = torch.einsum('ko,o->o', (next_layer.weight, outsigns)) #sign-weighted sum of next layer's weights per input neuron
                threshold = torch.log(torch.tensor(y.shape[1])).item()
                idx = torch.abs(influence) > threshold
                influence[idx] = threshold
                dw = torch.einsum('b,o,bi,bo->oi', (signal, influence, x, sgn)) #batched outer product of x, sgn. signal and influence are broadcasted
                db = torch.einsum('b,o,bo->o', (signal, influence, sgn)) #as above, but for bias
                linear_no += 1
                
                layer.weight += rate*dw / x.shape[0]
                layer.bias += rate*db / x.shape[0]  
                
                #scatterplots 
                inf = influence.cpu().detach().numpy()
                wandb.log({'influence': wandb.Histogram(inf)})
                
                sparsity = y > 0
                sparsity = sparsity.float().mean(dim=0).cpu().detach().numpy()
                #log sparsity per layer, but on the same plot
                wandb.log({'layer_{}_sparsity'.format(linear_no): wandb.Histogram(sparsity)})
                
                wandb.log({'layer_{}_weight'.format(linear_no): wandb.Histogram(layer.weight.cpu().detach().numpy())})
    
    def simple_kickback(self, signal, rate=0.1):
        device = os.environ['KICKBACK_DEVICE']
        for i, layer in enumerate(self.torso):
            if isinstance(layer, HebbianLinear):
                #kickback rule is dw_ij = (global error) *  (downstream influence of y_j) * (x_i) * sgn_j
                #sgn_j denotes whether y_j is a positive or negative unit, and also if the neuron fires or not
                x = layer.x
                y = layer.y

                sgn = (y > 0).float().to(device)
                
                dw = torch.einsum('b,bi,bo->oi', (signal, x, sgn)) 
                db = torch.einsum('b,bo->o', (signal, sgn))
                
                layer.weight += rate*dw / x.shape[0]
                layer.bias += rate*db / x.shape[0]  
    
    def oja_kickback(self, signal=None, rate=0.01):
        for i, layer in enumerate(self.torso):
            if isinstance(layer, HebbianLinear):
                x = layer.x #presynatpic activity
                y = layer.y #postsynaptic activity
                
                yw = torch.einsum('bo,oi->bi', (y, layer.weight))
                inner = x - yw
                if signal is None:
                    signal = torch.ones(x.shape[0]).to(x.device)
                dw = torch.einsum('bi,bo,b->oi', (inner, y, signal)) / x.shape[0]
                layer.weight += rate*dw
    
    def hebbnet_rule(self, rate=0.01): 
        device = os.environ['KICKBACK_DEVICE']
        for i, layer in enumerate(self.torso):
            if isinstance(layer, HebbianLinear):   
                layer.hebbnet(rate)
                
    def hebbian_rule(self, rate=0.01): 
        device = os.environ['KICKBACK_DEVICE']
        layer_no = 0
        for i, layer in enumerate(self.torso):
            if isinstance(layer, HebbianLinear):   
                layer.hebbian(rate)
                wandb.log({'weight_{}'.format(layer_no): wandb.Histogram(layer.weight.view(-1).cpu().detach().numpy())})
                wandb.log({'dw_avg_{}'.format(layer_no): wandb.Histogram(layer.dw_avg.view(-1).cpu().detach().numpy())})
                wandb.log({'weight_sparsity_{}'.format(layer_no): torch.mean((torch.abs(layer.weight) < 0.01).float())})
                wandb.log({'activation_sparsity_{}'.format(layer_no): torch.mean((torch.abs(layer.y) < 0.01).float())})
            if isinstance(layer, HebbianConv2d):   
                layer.hebbian(rate)
                layer_no += 1
                wandb.log({'filter_{}'.format(layer_no): wandb.Histogram(layer.filter.view(-1).cpu().detach().numpy())})
                wandb.log({'dw_avg_{}'.format(layer_no): wandb.Histogram(layer.dw_avg.view(-1).cpu().detach().numpy())})
                wandb.log({'weight_sparsity_{}'.format(layer_no): torch.mean((torch.abs(layer.filter) < 0.01).float())})
                wandb.log({'activation_sparsity_{}'.format(layer_no): torch.mean((torch.abs(layer.y) < 0.01).float())})
                
                
class HebbianLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, epsilon = 0.01, rectify=False, exp_avg=0.99, wta=True):
        super().__init__()
        device = os.environ['KICKBACK_DEVICE']
        self.weight = torch.Tensor(out_dim, in_dim).to(device)
        self.bias = torch.Tensor(out_dim).to(device)
        
        #exponential initialization
        self.weight = torch.nn.init.xavier_normal_(self.weight)
        self.bias = torch.nn.init.zeros_(self.bias)
        
        self.dw_avg = torch.zeros_like(self.weight).to(device)
        self.exp_avg = exp_avg
        self.wta = wta
        self.next = None
        
    def forward(self, x):
        self.x = x
        self.y = torch.einsum('oi,bi->bo', (self.weight, x)) + self.bias
        return self.y
    
    def softhebb(self, rate):
        x = self.x
        u = self.y #b, o
        y = F.softmax(u, dim=1)
        inner = (x - torch.matmul(u, self.weight)) #(b,o) * (o,i) = (b,i)
        dw = torch.einsum('bi,bo->oi', (inner, y)) / x.shape[0]
        
        self.weight += rate*dw
        
    def influencehebb(self, rate, softz, softy):
        x = self.x
        u = self.y #b, o
        
        if softy:
            y = F.softmax(u, dim=1)
        else:
            y = u
        
        inner = (x - torch.matmul(u, self.weight)) #(b,o) * (o,i) = (b,i)
        dw = torch.einsum('bi,bo->oi', (inner, y)) / x.shape[0]
        
        if self.next is None:
            print('ERROR: No next layer for influence hebbian')
            raise Exception
        
        if softz:
            z = F.softmax(self.next.y, dim=1)
        else:
            z = self.next.y
        
        influence = torch.einsum('bo,oi->bi', (z, self.next.weight)) #shape (b,o) where o is size of this layer output
        influence /= x.shape[0] #average over batch
        dw = torch.einsum('bo, oi->oi', (influence, dw))
            
        self.weight += rate*dw
        
       
#custom implementation of convolutional layer
class HebbianConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, epsilon = 0.01):
        super().__init__()
        self.filter = torch.randn(out_channels, in_channels, kernel_size, kernel_size).to(os.environ['KICKBACK_DEVICE'])
        self.bias = torch.randn(out_channels).to(os.environ['KICKBACK_DEVICE'])
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.epsilon = epsilon
        
        self.filter = torch.nn.init.xavier_normal_(self.filter)
        self.bias = torch.nn.init.zeros_(self.bias)
        
        self.dw_avg = torch.zeros_like(self.filter)
        self.wta = False
        
    def forward(self, x):
        self.x = x
        self.y = F.conv2d(x, self.filter, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return self.y
    
    def hebbian(self, rate=0.001):
        xpatches = F.unfold(self.x, self.kernel_size, self.stride, self.padding, self.dilation)
        ypatches = F.unfold(self.y, 1, 1, 0, 1)
        
        ysgn = (ypatches > 0).float()

        #hebbian learning rule: dw_ij = (x_i) * (y_j)
        dw = torch.einsum('bip,bop->oi', (xpatches, ysgn)) 
            
        dw = dw / xpatches.shape[0]
        dw = dw / xpatches.shape[1]
        dw = dw * rate
        dw = dw.view(self.filter.shape)
        dw_avg = 0.9*self.dw_avg + 0.1*dw
        dw -= self.dw_avg
        self.dw_avg = dw_avg
        
        if self.wta:
            #winner-take-all, top 10% of weights (in absolute value) are changed
            inhibition = 0 #inhibit all other weights
            n = int(dw.shape[0] * dw.shape[1] * dw.shape[2] * dw.shape[3] * 0.1)
            _, idx = torch.topk(torch.abs(dw.reshape(-1)), n, dim=0)
            mask = torch.zeros(dw.shape[0] * dw.shape[1] * dw.shape[2] * dw.shape[3]).to(os.environ['KICKBACK_DEVICE']) - inhibition
            mask[idx] = 1
            dw *= mask.reshape(dw.shape)
        
        self.filter += dw 
        
        
    def hebbian_pca(self, rate=0.001):
        xpatches = F.unfold(self.x, self.kernel_size, self.stride, self.padding, self.dilation).reshape(self.x.shape[0], self.in_channels, self.kernel_size * self.kernel_size, -1)
        ypatches = F.unfold(self.y, 1, 1, 0, 1).reshape(self.y.shape[0], self.out_channels, self.kernel_size * self.kernel_size, -1)
        actpatches = F.relu(ypatches)
        
        dw = torch.zeros_like(self.filter) #shape (out_channels, in_channels, kernel_size, kernel_size)
  
class MLP1(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, norm=False, activation=nn.ReLU) -> None:
        super().__init__()
        self.torso = torch.nn.Sequential()
        self.layers = []
        self.in_dim = in_dim
        for i in range(num_layers):
            layer = None
            norm = None
            if i == 0:
                if norm:
                    norm = torch.nn.LayerNorm(in_dim)
                layer = HebbianLinear(in_dim, hidden_dim)
            elif i == num_layers - 1:
                if norm:
                    norm = torch.nn.LayerNorm(hidden_dim)
                layer = HebbianLinear(hidden_dim, out_dim)
            else:
                if norm:
                    norm = torch.nn.LayerNorm(hidden_dim)
                layer = HebbianLinear(hidden_dim, hidden_dim)

            self.torso.add_module(f'layernorm_{i}', norm)
            self.torso.add_module(f'layer_{i}', layer)
            self.torso.add_module(f'activation_{i}', activation())
            
            self.layers += [layer]
            
            for i, layer in enumerate(self.layers):
                if i < len(self.layers) - 1:
                    layer.next = self.layers[i+1]
    def forward(self, x):
        return self.torso(x)
    
class HebbianOptimizer:
    def __init__(self, model, lr=0.001, learning_rule='softhebb', supervised=True, influencehebb_soft_y=True, influencehebb_soft_z=True):
        self.model = model
        self.lr = lr
        self.hebbianlayers = model.layers
        self.head = self.hebbianlayers[-1]
        
        if learning_rule == 'softhebb':
            self.learning_rule = self.softhebb
            if supervised:
                self.hebbianlayers = self.hebbianlayers[:-1]
        elif learning_rule == 'influencehebb':
            self.learning_rule = self.influencehebb
            self.hebbianlayers = self.hebbianlayers[:-1]
            self.influencehebb_soft_y = influencehebb_soft_y
            self.influencehebb_soft_y = influencehebb_soft_z
            
        elif learning_rule == 'random':
            self.learning_rule = self.random
            if not supervised:
                print('ERROR: random learning rule is not compatible with unsupervised learning')
                raise ValueError
        elif learning_rule == 'end2end':
            self.learning_rule = self.backprop
            if not supervised:
                print('ERROR: backprop learning rule is not compatible with unsupervised learning')
                raise ValueError
            
        self.supervised = supervised
        if supervised and not learning_rule == 'end2end':
            self.optim = torch.optim.SGD(self.head.parameters(), lr=lr)
        if supervised and learning_rule == 'end2end':
            self.optim = torch.optim.SGD(self.model.parameters(), lr=lr)
            
    
    def softhebb(self, rate=0.001):
        for layer in self.model.linears:
            layer.softhebb(rate)

        if self.supervised:
            self.head.backprop(rate)
        
    def influencehebb(self, rate=0.001):
        for layer in self.hebbianlayers:
            layer.hebbian_pca(rate)
        
        if self.supervised:
            self.head.backprop(rate)
        else:
            self.head.softhebb(rate)
    
    def random(self, rate=0.001):
        for layer in self.model.linears:
            layer.random(rate)
    
    def backprop(self, rate=0.001):
        pass
    
    def step(self):
        self.learning_rule(self.lr)
                  
        
def main(*args, **kwargs):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--learning_rule', type=str, default=None) #end-to-end backprop, softhebb, influencehebb, random
    parser.add_argument('--dataset', type=str, default='mnist') #mnist, cifar10, cifar100, ms_coco
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_type', type=str, default='mlp-1')
    parser.add_argument('--model_size', type=int, default=128)
    parser.add_argument('--model_depth', type=int, default=1)
    parser.add_argument('--supervised', type=int, default=True) #if false, train unsupervised and then fit linear probe
    parser.add_argument('--task', type=str, default='classification') #if dataset is ms coco, then we can do: a number of different tasks
    parser.add_argument("--influencehebb_soft_y", type=bool, default=True) #if true, use softmaxed y for influence hebbian learning rule
    parser.add_argument("--influencehebb_soft_z", type=bool, default=True) 
    parser.add_argument("--activation", type=str, default='relu') #relu, mish, triangle
    
    args = parser.parse_args()
    
    epochs = args.epochs
    lr = args.lr
    learning_rule = args.learning_rule
    assert learning_rule in ['end2end', 'softhebb', 'influencehebb', 'random']
    
    dataset = args.dataset
    assert dataset in ['mnist', 'cifar10', 'cifar100', 'ms_coco']
    
    batch_size = args.batch_size
    
    model = args.model_type
    assert model in ['mlp-1']
    
    width = args.model_size
    depth = args.model_depth
    
    supervised = args.supervised
    task = args.task
    
    influencehebb_soft_y = args.influencehebb_soft_y
    influencehebb_soft_z = args.influencehebb_soft_z
    
    act = args.activation
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['KICKBACK_DEVICE'] = str(device)
    
    print("Using device: ", device)
    model = None
    if model == 'mlp-1':
        model = MLP1(width, depth, act=act)
    
    net = HebbianNetwork(model, learning_rule=learning_rule, supervised=supervised, task=task, influencehebb_soft_y=influencehebb_soft_y, influencehebb_soft_z=influencehebb_soft_z)
    
    net = net.to(device)
    wandb.watch(net)
    
    train = None
    if dataset == 'mnist':
        train = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
        test = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar10':
        train = torch.utils.data.DataLoader(datasets.CIFAR10('data', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
        test = torch.utils.data.DataLoader(datasets.CIFAR10('data', train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar100':
        train = torch.utils.data.DataLoader(datasets.CIFAR100('data', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
        test = torch.utils.data.DataLoader(datasets.CIFAR100('data', train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
    elif dataset == 'ms_coco':
        if task == 'classification':
            train = torch.utils.data.DataLoader(datasets.CocoDetection('data', 'train2017', transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
            test = torch.utils.data.DataLoader(datasets.CocoDetection('data', 'val2017', transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
    
    
        
    epochs = 10

    optimizer = torch.optim.SGD(net.head.parameters(), lr=0.01)

    for epoch in range(epochs):
        print("Hebbian training, epoch: ", epoch+1, "/", epochs, "")
        #filterViz(net, epoch)
        viewNeurons(net, epoch)
        for batch in train:
            x, y = batch
            x = x.view(x.shape[0], -1).to(device)
            y = F.one_hot(y, num_classes=10).float().to(device)
            z = net.forward(x)
            loss = F.binary_cross_entropy(z, y, reduction='none')
            #reduce over classes
            loss = loss.mean(dim=1)
            net.hebbian_rule(rate=0.0)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({'train_loss': loss.item()})
            
        #get test loss and accuracy
        test_loss = 0
        test_acc = 0
        for batch in test:
            x, y = batch
            x = x.view(x.shape[0], -1).to(device)
            y = torch.nn.functional.one_hot(y, num_classes=10).float().to(device)
            z = net(x)
            loss = torch.nn.functional.binary_cross_entropy(z, y, reduction='none')
            loss = loss.mean(dim=1)
            loss = loss.mean()
            test_loss += loss.item()
            
            #plot confusion matrix to wandb
            z = z.argmax(dim=1)
            y = y.argmax(dim=1)
            
            test_acc += (z == y).float().mean().item()
            
        test_loss /= len(test)
        test_acc /= len(test)
        print("Test loss: ", test_loss, "Test accuracy: ", test_acc)
        wandb.log({'test_loss': test_loss, 'test_acc': test_acc})
        print("")
    
    
def filterViz(net, iter):
    if iter == 0:
        for f in os.listdir("./filters"):
            os.remove("./filters/"+f)
    
    weights = net.torso[1].filter.detach().cpu().numpy()
    #normalize rgb values to [0, 1]
    #plot  5x5 grid of neurons
    fig, axes = plt.subplots(5, 5)
    for i in range(5):
        for j in range(5):
            idx = i*5+j
            img = weights[idx].reshape(-1, 3 ,3)
            img = weights[idx] /(2*np.max(np.abs(weights[idx])))
            img = img + 0.5
            axes[i, j].imshow(img)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    plt.savefig("./filters/{}.png".format(iter))
    wandb.log({"filters": [wandb.Image("./filters/{}.png".format(iter))]})
    plt.close()

def viewNeurons(net, iter):    
    if iter == 0:
        for f in os.listdir("./neurons"):
            os.remove("./neurons/"+f)
    
    weights = net.torso[1].weight.detach().cpu().numpy()
    #plot  5x5 grid of neurons
    fig, axes = plt.subplots(5, 5)
    for i in range(5):
        for j in range(5):
            axes[i, j].imshow(weights[i*5+j].reshape(28, 28), cmap='gray', )
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    plt.savefig("./neurons/{}.png".format(iter))
    plt.close()

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
    