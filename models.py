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