import torch

class HebbianOptimizer:
    def __init__(self, model, lr=0.001, learning_rule='softhebb', supervised=True, influencehebb_soft_y=True,
                 influencehebb_soft_z=True, adaptive_lr=False, adaptive_lr_p=1, schedule=None):
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
            self.influencehebb_soft_z = influencehebb_soft_z
        elif learning_rule == 'random':
            self.learning_rule = self.random
            if not supervised:
                raise Exception('random learning rule is not compatible with unsupervised learning')
        elif learning_rule == 'end2end':
            self.learning_rule = self.backprop
            if not supervised:
                raise Exception('end2end learning rule is not compatible with unsupervised learning')
        else:
            raise Exception('learning rule not recognized')
            
        self.supervised = supervised
        if supervised and not learning_rule == 'end2end':
            self.optim = torch.optim.SGD(self.head.parameters(), lr=lr)
        if supervised and learning_rule == 'end2end':
            self.optim = torch.optim.SGD(self.model.parameters(), lr=lr)
            
        self.adaptive_lr = adaptive_lr
        self.p = adaptive_lr_p
            
    
    def softhebb(self, rate=0.001):
        with torch.no_grad():
            for i, layer in enumerate(self.hebbianlayers):
                dw = layer.softhebb(rate, adaptive=self.adaptive_lr, p=self.p)
                

        if self.supervised:
            self.optim.step()
            self.optim.zero_grad()
        
    def influencehebb(self, rate=0.001):
        with torch.no_grad():
            for i, layer in enumerate(self.hebbianlayers):
                dw = layer.influencehebb(rate, self.influencehebb_soft_y, self.influencehebb_soft_z, adaptive=self.adaptive_lr, p=self.p)
        
        if self.supervised:
            self.optim.step()
            self.optim.zero_grad()
        else:
            with torch.no_grad():
                self.head.softhebb(rate, adaptive_lr=self.adaptive_lr, p=self.p)
    
    def random(self, rate=0.001):
        self.optim.step()
        self.optim.zero_grad()
    
    def backprop(self, rate=0.001):
        pass
    
    def step(self):
        self.learning_rule(self.lr)
    