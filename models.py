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
    def __init__(self, img_dim, img_chan, hidden_chan, out_dim, num_layers, activation=nn.ReLU, normlayer=None, poollayer=None, init=None, init_radius=None) -> None:
        super().__init__()
        self.torso = torch.nn.Sequential()
        self.layers = []
        
        self.torso.add_module('flatten', nn.Flatten())
        curr_dim = img_dim
        for i in range(num_layers):
            layer = None
            norm = None
            act = activation()
            
            if curr_dim % 2 == 0 and curr_dim > 4 and poollayer:
                pool = poollayer(2)
                curr_dim = curr_dim // 2
            else:
                pool = None
            
            if i == 0:
                if normlayer:
                    norm = normlayer(img_chan)
                layer = HebbianConv2d(img_chan, hidden_chan, 3, stride=1, padding=1, init=init, init_radius=init_radius, act=act)
            elif i == num_layers - 1:
                if normlayer:
                    norm = normlayer(hidden_chan)
                layer = HebbianConv2d(hidden_chan, hidden_chan, 3, stride=1, padding=1, init=init, init_radius=init_radius, act=nn.Softmax(dim=1))
            else:
                if normlayer:
                    norm = normlayer(hidden_chan)
                layer = HebbianConv2d(hidden_chan, hidden_chan, 3, stride=1, padding=1, init=init, init_radius=init_radius, act=act)

            if norm:
                self.torso.add_module(f'layernorm_{i}', norm)
            self.torso.add_module(f'layer_{i}', layer)
            if pool:
                self.torso.add_module(f'pool_{i}', pool)
            
            self.layers += [layer]
        
        head = nn.Linear(curr_dim*curr_dim*hidden_chan, out_dim)
        self.torso.add_module('flatten', nn.Flatten())
        self.torso.add_module('head', head)
        self.torso.add_module('softmax', nn.Softmax(dim=1))
        
        self.layers += [head]
        
            
        
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                layer.next = self.layers[i+1]
        
        
            
        
                    
    def forward(self, x):
        x = self.torso(x)
        return x