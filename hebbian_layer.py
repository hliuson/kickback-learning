import torch
import torch.nn as nn
import os
import wandb
import torch.nn.functional as F


def _hebb(x,u,y, weight, rate, adaptive, p, influence=None, dot_uw=False):
    if dot_uw:
        uw = torch.matmul(u, weight) #(b, o) x (o, i) = (b, i)
        uw = uw.view(-1, 1, uw.shape[-1]) #shape (b, 1, i)
    else:
        u = u.view(-1, u.shape[-1], 1) #shape (b, o, 1)
        uw = u * weight #not matmul since u is broadcasted. shape (b, o, i)
    x = x.view(-1, 1, x.shape[-1]) #shape (b, 1, i)
    inner = x - uw #shape (b, o, i)
    y = y.view(-1, y.shape[-1], 1) #shape (b, o, 1)
    dw = inner * y #shape (b, o, i)
    
    norm = torch.norm(weight, dim=1)
        
    if adaptive:
        diff = torch.abs(1 - norm)
        rate = rate * torch.pow(diff, p)  
        rate = rate.view(-1,1)

    if influence is not None:
        #center and normalize influence
        dw = dw * influence
        #wandb.log({'influence': torch.mean(torch.abs(influence))})
    
    #reduce over batch
    dw = torch.mean(dw, dim=0) #shape (o, i)
    return rate*dw

class HebbianLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, epsilon = 0.01, rectify=False, exp_avg=0.99, wta=True, hebbian=True, init='xavier', init_radius=None, act = None):
        super().__init__()
        device = os.environ['KICKBACK_DEVICE']
        self.weight = torch.Tensor(out_dim, in_dim).to(device)
        self.bias = torch.Tensor(out_dim).to(device)
        
        R = init_radius
        
        if init == 'normal':
            N = in_dim
            std = ((torch.pi /(2*N)) ** 0.5) * R
            self.weight = torch.nn.init.normal_(self.weight, std=std)
            self.bias = torch.nn.init.zeros_(self.bias)
        
            
        if init == 'positive-uniform':
            N = in_dim
            range = ((1 / N) ** 0.5) * R * 2
            self.weight = torch.nn.init.uniform_(self.weight, a=0, b=range)
            self.bias = torch.nn.init.zeros_(self.bias)
            
        if init == 'xavier':
            self.weight = torch.nn.init.xavier_normal_(self.weight, gain=R)
            self.bias = torch.nn.init.zeros_(self.bias)
            
        
        self.dw_avg = torch.zeros_like(self.weight).to(device)
        self.exp_avg = exp_avg
        self.wta = wta
        self.next = None
        
        self.weight = torch.nn.Parameter(self.weight)
        self.bias = torch.nn.Parameter(self.bias)
        
        self.hebbian = hebbian
        self.act = act
        if act is None:
            self.act = torch.nn.Identity()
        
    def forward(self, x):
        self.x = x
        self.u = torch.matmul(self.weight, x.t()).t() + self.bias
        self.y = self.act(self.u)
        return self.y
    
    def softhebb(self, rate, adaptive=False, p=0.5, dot_uw=False):
        x = self.x
        u = self.u #b, o
        y = F.softmax(self.y, dim=1)
        
        step = _hebb(x, u, y, self.weight, rate, adaptive, p, dot_uw=dot_uw)
        self.weight += step
        wandb.log({'step size': torch.mean(torch.abs(step))}, commit=False)
        
        
    def influencehebb(self, rate, softz, softy, adaptive=False, p=0.5, influence_type='simple', dot_uw=False):
        x = self.x
        u = self.u #b, o
        y = self.y
        if softy:
            y = F.softmax(y, dim=1)
        
        
        z = self.next.u #or self.next.y?
        if softz:
            z = F.softmax(z, dim=1)

        #influence = torch.matmul(z, self.next.weight) #shape (b, o)
        #influence = influence.view(-1, influence.shape[1], 1) #shape (b, o, 1)
    
        if influence_type == 'grad':
            pre = self.y
            post = self.next.u
            influence = torch.autograd.grad(post, pre, grad_outputs=z, retain_graph=True)[0]
        if influence_type == 'simple':
            influence = torch.matmul(z, self.next.weight) #shape (b, o)
        if influence_type == 'one':
            influence = torch.ones_like(self.y)
            
        influence = influence.view(-1, influence.shape[1], 1) #shape (b, o, 1)
        
        step = _hebb(x, u, y, self.weight, rate, adaptive, p, influence, dot_uw=dot_uw)
        self.weight += step
        wandb.log({'step size': torch.mean(torch.abs(step))}, commit=False)
        
        
        
       
#custom implementation of convolutional layer
class HebbianConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, epsilon = 0.01, act = None):
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
        self.act = act
        if act is None:
            self.act = torch.nn.Identity()
        
    def forward(self, x):
        self.x = x
        self.u = F.conv2d(x, self.filter, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.y = self.act(self.u)
        return self.y
            
    def softhebb(self, rate, adaptive=False, p=0.5):
        x = F.unfold(self.x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)
        u = F.unfold(self.y, kernel_size=1, stride=1, padding=0)
        y = F.softmax(u, dim=1) #softmax over output neurons per patch / batch
        
        self.filter += _hebb(x, u, y, self.filter, rate, adaptive, p)
        
    def influencehebb(self, rate, softz, softy, adaptive=False, p=0.5, influence_type='simple', dot_uw=False):
        x = self.x
        u = self.u #b, o
        y = self.y
        if softy:
            y = F.softmax(y, dim=1)
        
        
        z = self.next.u #or self.next.y?
        if softz:
            z = F.softmax(z, dim=1)

        #influence = torch.matmul(z, self.next.weight) #shape (b, o)
        #influence = influence.view(-1, influence.shape[1], 1) #shape (b, o, 1)
    
        if influence_type == 'grad':
            pre = self.y
            post = self.next.u
            influence = torch.autograd.grad(post, pre, grad_outputs=z, retain_graph=True)[0]
        else:
            raise NotImplementedError()
            
        influence = influence.view(-1, influence.shape[1], 1) #shape (b, o, 1)
        
        self.weight += _hebb(x, u, y, self.weight, rate, adaptive, p, influence, dot_uw=dot_uw)
        
        
  