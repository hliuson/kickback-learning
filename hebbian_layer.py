import torch
import torch.nn as nn
import os
import wandb
import torch.nn.functional as F
from abc import ABC, abstractmethod

class SofthebbArgs:
    def __init__(self, rate, adaptive, p, influence=None, dot_uw=False, temp=1):
        self.rate = rate
        self.adaptive = adaptive
        self.p = p
        self.temp = temp
        self.dot_uw = dot_uw
        
class InfluenceArgs:
    def __init__(self, rate, adaptive, p, dot_uw=False, softz = True, softy = True, influence_type='grad', temp=1):
        self.rate = rate
        self.adaptive = adaptive
        self.p = p
        self.dot_uw = dot_uw
        self.softz = softz
        self.softy = softy
        self.temp = temp
        self.influence_type = influence_type
class HebbianLayer(ABC):
    @abstractmethod
    def softhebb(self, args:SofthebbArgs):
        pass
    
    @abstractmethod
    def influencehebb(self, args:InfluenceArgs):
        pass
    
    @abstractmethod
    def clear_grad(self):
        pass
    
    @abstractmethod
    def output(self):
        pass
    
    @abstractmethod
    def getweight(self):
        pass

    def getnext(self):
        print('getnext called, self.next: ', self.next)
        return self.next

def _hebb(x,u,y, weight, rate, adaptive, p, influence=None, dot_uw=False, batch_size=256):
    #batch/patch number is very large so we chunk it
    num_batches = x.shape[0] // batch_size
    if x.shape[0] % batch_size > 0:
        num_batches += 1

    dw_accum = []  
    for batch_idx in range(num_batches):
        xb = x[batch_idx*batch_size:(batch_idx+1)*batch_size]
        ub = u[batch_idx*batch_size:(batch_idx+1)*batch_size]
        yb = y[batch_idx*batch_size:(batch_idx+1)*batch_size]
        if influence is not None:
            ib = influence[batch_idx*batch_size:(batch_idx+1)*batch_size]
        if dot_uw:
            uw = torch.matmul(ub, weight) #(b, o) x (o, i) = (b, i)
            uw = uw.view(-1, 1, uw.shape[-1]) #shape (b, 1, i)
        else:
            ub = ub.view(-1, ub.shape[-1], 1) #shape (b, o, 1)
            uw = ub * weight #not matmul since u is broadcasted. shape (b, o, i)
        xb = xb.view(-1, 1, xb.shape[-1]) #shape (b, 1, i)
        inner = xb - uw #shape (b, o, i)
        yb = yb.view(-1, yb.shape[-1], 1) #shape (b, o, 1)
        dw = inner * yb #shape (b, o, i)
        
        if influence is not None:
            #center and normalize influence
            dw = dw * ib 
        
        dw_accum.append(dw)
    dw = torch.cat(dw_accum, dim=0) 
    del dw_accum, xb, ub, yb, uw, inner, ib
        #wandb.log({'log/weightnorm': torch.mean(norm)}, commit=False)
            
    if adaptive:
        norm = torch.norm(weight, dim=1)
        diff = torch.abs(1 - norm)
        rate = rate * torch.pow(diff, p)  
        rate = rate.view(-1,1)

        #wandb.log({'influence': torch.mean(torch.abs(influence))})
    
    #reduce over batch
    dw = torch.mean(dw, dim=0) #shape (o, i)
    step = rate * dw
    #wandb.log({'log/step size': torch.mean(torch.abs(step))}, commit=False)
    return step

def _influencehebb(x: torch.Tensor,u:torch.Tensor,y: torch.Tensor, self: HebbianLayer, next: HebbianLayer, args: InfluenceArgs):
        influence_type = args.influence_type
            
        z = next.output()
        if args.softz:
            z = torch.softmax(z / args.temp, dim=1)
            
        if influence_type == 'grad':
            pre = self.output()
            post = next.output()
            influence = torch.autograd.grad(post, pre, grad_outputs=z, retain_graph=True)[0]

        if influence_type == 'simple':
            raise NotImplementedError
        if influence_type == 'one':
            raise NotImplementedError
        influence = influence.view(-1, influence.shape[1], 1) #shape (b, o, 1)
        
        if args.softy:
            y = torch.softmax(y / args.temp, dim=1)

        
        step = _hebb(x, u, y, self.getweight(), args.rate, args.adaptive, args.p, influence, dot_uw=args.dot_uw)
        return step
    

class HebbianLinear(torch.nn.Module, HebbianLayer):
    def __init__(self, in_dim, out_dim, epsilon = 0.01, rectify=False, exp_avg=0.99, wta=True, hebbian=True, init='xavier', init_radius=None, act = None):
        super().__init__()
        device = os.environ['KICKBACK_DEVICE']
        self.weight = torch.Tensor(out_dim, in_dim).to(device)
        self.bias = torch.Tensor(out_dim).to(device)
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        R = init_radius
        
        self.init_params(init, R)
            
        self.next = None
        
        self.weight = torch.nn.Parameter(self.weight)
        self.bias = torch.nn.Parameter(self.bias)
        
        self.hebbian = hebbian
        self.act = act
        if act is None:
            self.act = torch.nn.Identity()

    def init_params(self, init="xavier", init_radius="1"):
        R = init_radius
        N = self.in_dim
        if init == 'normal':
            std = ((torch.pi /(2*N)) ** 0.5) * R
            self.weight = torch.nn.init.normal_(self.weight, std=std)
            self.bias = torch.nn.init.zeros_(self.bias)
        
            
        if init == 'positive-uniform':
            range = ((1 / N) ** 0.5) * R * 2
            self.weight = torch.nn.init.uniform_(self.weight, a=0, b=range)
            self.bias = torch.nn.init.zeros_(self.bias)
            
        if init == 'xavier':
            self.weight = torch.nn.init.xavier_normal_(self.weight, gain=R)
            self.bias = torch.nn.init.zeros_(self.bias)
        
    def forward(self, x):
        self.x = x
        self.u = torch.matmul(self.weight, x.t()).t() + self.bias
        self.y = self.act(self.u)
        return self.y
    
    def softhebb(self, args:SofthebbArgs):
        temp = args.temp
        rate = args.rate
        adaptive = args.adaptive
        p = args.p
        dot_uw = args.dot_uw
        x = self.x
        u = self.u #b, o
        y = F.softmax(self.y / temp, dim=1)
        
        step = _hebb(x, u, y, self.weight, rate, adaptive, p, dot_uw=dot_uw)
        self.weight += step
        
        
    def influencehebb(self, args: InfluenceArgs):
        x = self.x
        u = self.u #b, o
        y = self.y
        
        step = _influencehebb(x, u, y, self, self.next, args)
        
    def clear_grad(self):
        self.weight.grad = None
        self.bias.grad = None
        
        self.x = None
        self.y = None
        self.z = None
        
    def output(self): 
        return self.y
    
    def getweight(self):
        return self.weight
        
        
       
#custom implementation of convolutional layer
class HebbianConv2d(torch.nn.Module, HebbianLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, init='xavier', init_radius=None, act = None):
        super().__init__()
        device = os.environ['KICKBACK_DEVICE']
        self.filter = torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).to(device)
        self.bias = torch.Tensor(out_channels).to(device)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.init_params(init, init_radius)

        self.act = act
        if act is None:
            self.act = torch.nn.Identity()

    def init_params(self, init="xavier", init_radius="1"):
        R = init_radius
        N = self.in_channels
        if init == 'normal':
            std = ((torch.pi /(2*N)) ** 0.5) * R
            self.filter = torch.nn.init.normal_(self.filter, std=std)
            self.bias = torch.nn.init.zeros_(self.bias)
        
            
        if init == 'positive-uniform':
            range = ((1 / N) ** 0.5) * R * 2
            self.filter = torch.nn.init.uniform_(self.filter, a=0, b=range)
            self.bias = torch.nn.init.zeros_(self.bias)
            
        if init == 'xavier':
            self.filter = torch.nn.init.xavier_normal_(self.filter, gain=R)
            self.bias = torch.nn.init.zeros_(self.bias)

        
    def forward(self, x):
        self.x = x
        self.u = F.conv2d(x, self.filter, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.y = self.act(self.u)
        return self.y
            
    def softhebb(self, args:SofthebbArgs):
        temp = args.temp
        rate = args.rate
        adaptive = args.adaptive
        p = args.p
        x = F.unfold(self.x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)
        u = self.u
        
         #!!! VERY IMPORTANT. Spatial dimensions must be LEFT of channel dimension before flattening.
        u = u.permute(0, 3, 2, 1) #permute to (b, o, h, w)
        u = u.reshape(-1, self.out_channels)
        y = F.softmax(u / temp, dim=1) #softmax over output channels
        
        x = x.permute(0, 2, 1) #permute to (b, h*w, i*k*k)
        x = x.reshape(-1, self.in_channels*self.kernel_size*self.kernel_size) #reshape to (b*h*w, i*k*k)
                        
        filter = self.filter.view(self.out_channels, -1)
        step = _hebb(x, u, y, filter, rate, adaptive, p).view(self.filter.shape)
        self.filter += step
        
    def influencehebb(self, args: InfluenceArgs):
        temp = args.temp
        softz = args.softz
        influence_type = args.influence_type
        rate = args.rate
        adaptive = args.adaptive
        p = args.p
        dot_uw = args.dot_uw
        
        x = F.unfold(self.x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)
        u = self.u
        
         #!!! VERY IMPORTANT. Spatial dimensions must be LEFT of channel dimension before flattening.
        u = u.permute(0, 3, 2, 1) #permute to (b, o, h, w)
        u = u.reshape(-1, self.out_channels)
        y = F.softmax(u / temp, dim=1) #softmax over output channels
        
        x = x.permute(0, 2, 1) #permute to (b, h*w, i*k*k)
        x = x.reshape(-1, self.in_channels*self.kernel_size*self.kernel_size) #reshape to (b*h*w, i*k*k)
        
        
        z = self.next.output()
        if softz:
            z = F.softmax(z / temp, dim=1) #softmax over output neurons
    
        if influence_type == 'grad':
            pre = self.y #shape (b, c, h, w)
            post = self.next.output()
            influence = torch.autograd.grad(post, pre, grad_outputs=z, retain_graph=True)[0]
            influence = F.unfold(influence, kernel_size=1, stride=1, padding=0)
        else:
            raise NotImplementedError()
            
        influence = influence.permute(0, 2, 1) #permute to (b, h*w, o) 
        influence = influence.reshape(-1, self.out_channels, 1) #reshape to (b*h*w, o)
        
        filter = self.filter.view(self.out_channels, -1)
        step = _hebb(x, u, y, filter, rate, adaptive, p, influence, dot_uw=dot_uw).view(self.filter.shape)
        self.filter += step
        
    def clear_grad(self):
        self.filter.grad = None
        self.bias.grad = None
        
        self.x = None
        self.y = None
        self.z = None
        
    def output(self):
        return self.y
    
    def getweight(self):
        return self.filter
        
class SkipBlock(torch.nn.Module, HebbianLayer):
    def __init__(self, in_dim, out_dim, init=None, init_radius=None, act=nn.ReLU(), skip=True):
        super().__init__()
        self.layer = HebbianLinear(in_dim, out_dim, init=init, init_radius=init_radius, act=act)
        self.main_path = nn.Sequential(
            self.layer,
        )
        
        self.skip = skip

        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        y = self.main_path(x)
        
        if not self.skip:
            self.y = y
            return self.y
        
        if self.in_dim > self.out_dim:
            z = x[:, :self.out_dim]
        
        if self.in_dim < self.out_dim:
            z = torch.cat((x, torch.zeros((x.shape[0], self.in_dim - self.out_dim), device=x.device)), dim=1)
            
        if self.in_dim == self.out_dim:
            z = x
        
        self.y = y + z
        return self.y

    def linear(self):
        return self.layer
    
    def softhebb(self, args: SofthebbArgs):
        self.layer.softhebb(args)
    
    def influencehebb(self, args: InfluenceArgs):
        step = _influencehebb(self.layer.x, self.layer.u, self.layer.y, self, self.next, args)
        self.layer.weight += step

    def output(self):
        return self.y
    
    def getweight(self):
        return self.layer.weight
    
    def clear_grad(self):
        return self.layer.clear_grad()