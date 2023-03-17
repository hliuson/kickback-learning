import torch
import os
import wandb
import torch.nn.functional as F


class HebbianLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, epsilon = 0.01, rectify=False, exp_avg=0.99, wta=True, hebbian=True, init='xavier', init_radius=None):
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
        
    def forward(self, x):
        self.x = x
        self.y = torch.matmul(self.weight, x.t()).t() + self.bias
        return self.y
    
    def softhebb(self, rate, adaptive=False, p=2):
        x = self.x
        u = self.y #b, o
        y = F.softmax(u, dim=1)
        inner = (x - torch.matmul(u, self.weight)) #(b,o) * (o,i) = (b,i)
        dw = torch.einsum('bi,bo->oi', (inner, y)) / x.shape[0]
        
         #scale dw by norm of weight vector
        norm = torch.norm(self.weight, dim=1)
        wandb.log({'avg weight norm': torch.mean(norm)})
            
        if adaptive:
            diff = torch.abs(1 - norm)
            #raise to power p
            rate = rate * torch.pow(diff, p)  
            wandb.log({'avg effective rate': torch.mean(rate)})
            rate = rate.view(-1,1)
        self.weight += rate*dw
        wandb.log({'avg dw': torch.mean(torch.abs(dw))})
        
    
        return dw.to('cpu').detach().numpy()
        
    def influencehebb(self, rate, softz, softy, adaptive=False, p=2):
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
            
        if adaptive:
            #scale dw by norm of weight vector
            norm = torch.norm(self.weight, dim=1)
            wandb.log({'avg weight norm': torch.mean(norm)})
            diff = torch.abs(1 - norm)
            #raise to power p
            rate = rate * torch.pow(diff, p)  
            wandb.log({'avg effective rate': torch.mean(rate)})
            rate = rate.view(-1,1)

        self.weight += rate*dw
        wandb.log({'avg dw': torch.mean(dw)})
        
        
        return dw.to('cpu').detach().numpy()
        
       
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
        
  