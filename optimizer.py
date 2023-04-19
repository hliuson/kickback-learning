import torch
import wandb
from hebbian_layer import *
class HebbianOptimizer:
    def __init__(self, model, lr=0.001, learning_rule='softhebb', supervised=True, influencehebb_soft_y=True,
                 influencehebb_soft_z=True, adaptive_lr=False, adaptive_lr_p=1, schedule=None, influence_type='simple',
                 dot_uw=False, temp=1, const_feedback=False, exp_avg=0.95, group_size=-1, shuffle=False, sequentialTraining=False):
        self.model = model
        self.lr = lr
        self.hebbianlayers = model.layers
        self.head = model.head
        
        if learning_rule == 'softhebb':
            self.learning_rule = self.softhebb
            if supervised:
                self.hebbianlayers = self.hebbianlayers[:-1]
            self.args = SofthebbArgs(
                rate = lr,
                adaptive=adaptive_lr,
                p = adaptive_lr_p,
                dot_uw=dot_uw,
                temp = temp
            )
        elif learning_rule == 'influencehebb':
            self.learning_rule = self.influencehebb
            self.hebbianlayers = self.hebbianlayers[:-1]
            
            self.args = InfluenceArgs(
                rate = lr,
                adaptive=adaptive_lr,
                p=adaptive_lr_p,
                dot_uw=dot_uw,
                softy=influencehebb_soft_y,
                softz=influencehebb_soft_z,
                influence_type=influence_type,
                temp = temp,
                const_feedback=const_feedback
            )
            
        elif learning_rule == 'random':
            self.learning_rule = self.random
            if not supervised:
                raise Exception('random learning rule is not compatible with unsupervised learning')
        elif learning_rule == 'end2end':
            self.learning_rule = self.backprop
            if not supervised:
                raise Exception('end2end learning rule is not compatible with unsupervised learning')
        elif learning_rule == 'simplesofthebb':
            self.learning_rule = self.simplesofthebb
            self.hebbianlayers = self.hebbianlayers[:-1]
            self.args = SimpleSofthebbArgs(
                rate = lr,
                temp = temp,
                exp_avg=exp_avg,
                group_size=group_size,
                shuffle=shuffle
            )
        elif learning_rule == 'kickback':
            self.learning_rule = self.kickback
            self.hebbianlayers = self.hebbianlayers[:-1]
            self.args = KickbackArgs(
                rate = lr,
            )
        else:
            raise Exception(f'learning rule: {learning_rule} not recognized')
            
        self.supervised = supervised
        if supervised and not learning_rule == 'end2end':
            self.optim = torch.optim.AdamW(self.head.parameters())
        if supervised and learning_rule == 'end2end':
            self.optim = torch.optim.AdamW(self.model.parameters())
            
        self.adaptive_lr = adaptive_lr
        self.p = adaptive_lr_p
        self.influence_type = influence_type
        self.dot_uw = dot_uw
        self.temp = temp
        
        if schedule == 'constant':
            self.schedule = self.const_schedule
        elif schedule == 'exponential':
            self.schedule = self.exp_schedule
        else:
            raise Exception(f'schedule: {schedule} not recognized')
        
        self.epoch = 0
        self.n_steps = 0
        self.sequentialTraining = sequentialTraining
        
    def const_schedule(self):
        return self.lr
    
    def exp_schedule(self):
        #TODO: make this more general
        return self.lr * 0.95 ** (self.n_steps / 100)
    
    def softhebb(self, rate=0.001):
        self.args.rate = rate
        with torch.no_grad():
            for i, layer in enumerate(self.hebbianlayers):
                layer.softhebb(self.args)
                #wandb.log({f'layer_{i}_weightnorm': torch.mean(torch.norm(layer.weight.data, dim=1))}, commit=False)
                
        if self.supervised:
            self.optim.step()
        self.clear_grad()

    def influencehebb(self, rate=0.001):
        self.args.rate = rate
        with torch.no_grad():
            for i, layer in enumerate(self.hebbianlayers):
                layer.influencehebb(self.args)
                #wandb.log({f'layer_{i}_weightnorm': torch.mean(torch.norm(layer.weight.data, dim=1))}, commit=False)
        
        if self.supervised:
            self.optim.step()
        else:
            with torch.no_grad():
                raise NotImplementedError
                self.head.softhebb(rate, adaptive=self.adaptive_lr, p=self.p)
        self.clear_grad()

    def simplesofthebb(self, rate=0.001):
        self.args.rate = rate
        with torch.no_grad():
            for i, layer in enumerate(self.hebbianlayers):
                if self.sequentialTraining:
                    if self.epoch == i: #train each layer sequentially for one epoch
                        layer.simplesofthebb(self.args)
                else:
                    layer.simplesofthebb(self.args)
                #wandb.log({f'layer_{i}_weightnorm': torch.mean(torch.norm(layer.weight.data, dim=1))}, commit=False)
        if self.supervised:
            self.optim.step()
        self.clear_grad()

    def kickback(self, rate=0.001):
        self.args.rate = rate
        with torch.no_grad():
            for i, layer in enumerate(self.hebbianlayers):
                layer.kickback(self.args)
                #wandb.log({f'layer_{i}_weightnorm': torch.mean(torch.norm(layer.weight.data, dim=1))}, commit=False)
        
        if self.supervised:
            self.optim.step()
        self.clear_grad()
    
    def random(self, rate=0.001):
        self.optim.step()
        self.optim.zero_grad()
    
    def backprop(self, rate=0.001):
        self.optim.step()
        self.optim.zero_grad()
    
    def step(self):
        lr = self.schedule()
        wandb.log({'lr': lr}, commit=False)
        self.learning_rule(lr)
        self.n_steps += 1
        
    def inc_epoch(self):
        self.epoch += 1
    
    def clear_grad(self):
        if self.supervised:
            self.optim.zero_grad()
        for layer in self.hebbianlayers:
            layer.clear_grad()