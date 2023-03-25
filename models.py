from hebbian_layer import *
import torch 
import torch.nn as nn

class MLP1(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, activation=nn.ReLU, normlayer=None, init=None, init_radius=None) -> None:
        super().__init__()
        self.torso = torch.nn.Sequential()
        self.layers = []
        
        self.torso.add_module('flatten', nn.Flatten())
        
        for i in range(num_layers):
            layer = None
            norm = None
            act = activation()
            if i == 0:
                if normlayer:
                    norm = normlayer(in_dim)
                layer = HebbianLinear(in_dim, hidden_dim, init=init, init_radius=init_radius, act=act)
            elif i == num_layers - 1:
                if normlayer:
                    norm = normlayer(hidden_dim)
                layer = HebbianLinear(hidden_dim, out_dim, init=init, init_radius=init_radius, act=nn.Softmax(dim=1))
            else:
                if normlayer:
                    norm = normlayer(hidden_dim)
                layer = HebbianLinear(hidden_dim, hidden_dim, init=init, init_radius=init_radius, act=act)

            if norm:
                self.torso.add_module(f'layernorm_{i}', norm)
            self.torso.add_module(f'layer_{i}', layer)
            
            self.layers += [layer]
            
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                layer.next = self.layers[i+1]
        
                    
    def forward(self, x):
        x = self.torso(x)
        return x
    
class CNN1(torch.nn.Module):
    def __init__(self, img_dim, img_chan, hidden_chan, out_dim, num_layers,
                 activation=nn.ReLU, normlayer=None, poollayer=None, init=None, init_radius=None,
                 pool_chan_mult=2) -> None:
        super().__init__()
        self.torso = torch.nn.Sequential()
        self.layers = []
        self.torso.add_module('reshape_1', nn.Flatten())
        self.torso.add_module('reshape_2', nn.Unflatten(1, (img_chan, img_dim, img_dim)))
        curr_dim = img_dim
        curr_chan = img_chan
        next_chan = hidden_chan
        for i in range(num_layers):
            layer = None
            norm = None
            act = activation()

            if curr_dim % 2 == 0 and curr_dim > 4 and poollayer:
                pool = poollayer(2)
                curr_dim = curr_dim // 2
                if i != 0:
                    next_chan *= pool_chan_mult
            else:
                pool = None   
            
            if normlayer:
                norm = normlayer(curr_chan)
            layer = HebbianConv2d(curr_chan, next_chan, 3, stride=1, padding=1, init=init, init_radius=init_radius, act=act)
            
            if norm:
                self.torso.add_module(f'layernorm_{i}', norm)
            self.torso.add_module(f'layer_{i}', layer)
            if pool:
                self.torso.add_module(f'pool_{i}', pool)
            
            self.layers += [layer]
            curr_chan = next_chan

        head = HebbianLinear(curr_dim*curr_dim*curr_chan, out_dim, init=init, init_radius=init_radius, act=nn.Softmax(dim=1))
        self.torso.add_module('flatten', nn.Flatten())
        self.torso.add_module('head', head)
        
        self.layers += [head]
        
        
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                layer.next = self.layers[i+1]
                    
    def forward(self, x):
        x = self.torso(x)
        return x