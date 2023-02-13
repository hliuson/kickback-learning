import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import os

def kickback_get_signs_for_length(length: int):
        #first half of the signs are positive, second half are negative]
        ones = torch.ones(length)
        ones[length//2:] = -1
        return ones
    
class PNReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.relu(x) * kickback_get_signs_for_length(x.shape[1]).to(x.device)

class KickbackNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, activation=PNReLU):
        super().__init__()
        self.torso = torch.nn.Sequential()
        self.linears = []
        for i in range(num_layers):
            linear = None
            if i == 0:
                linear = HebbianLinear(in_dim, hidden_dim)
            else:
                linear = HebbianLinear(hidden_dim, hidden_dim)
            self.torso.add_module(f'linear_{i}', linear)
            self.torso.add_module(f'activation_{i}', activation())
            self.torso.add_module(f'bn_{i}', torch.nn.BatchNorm1d(hidden_dim))
            
            self.linears += [linear]
            
        self.head = torch.nn.Sequential()
        linear = torch.nn.Linear(hidden_dim, out_dim)
        self.head.add_module('linear', linear)
        self.head.add_module('activation', torch.nn.Sigmoid())
        
        self.linears += [linear]
        
    def forward(self, x, endtoend=False):
        if endtoend:
            x = self.torso(x)
        else:
            with torch.no_grad():
                x = self.torso(x)
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
                dw = torch.einsum('b,o,bi,bo->oi', (signal, influence, x, sgn)) #batched outer product of x, sgn. signal and influence are broadcasted
                db = torch.einsum('b,o,bo->o', (signal, influence, sgn)) #as above, but for bias
                linear_no += 1
                
                layer.weight += rate*dw / x.shape[0]
                layer.bias += rate*db / x.shape[0]        
                
                
class HebbianLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        device = os.environ['KICKBACK_DEVICE']
        self.weight = torch.Tensor(out_dim, in_dim).to(device)
        self.bias = torch.Tensor(out_dim).to(device)
        
        #exponential initialization
        self.weight = torch.nn.init.xavier_normal_(self.weight)
        self.bias = torch.nn.init.zeros_(self.bias)
        
        #rectify signs
        #with torch.no_grad():
            #signs = kickback_get_signs_for_length(out_dim).to(device)
            #self.weight = torch.abs(self.weight) * signs.view(-1, 1)
        
        #self.weight.requires_grad = False
        #self.bias.requires_grad = False
        
    def forward(self, x):
        self.x = x
        self.y = torch.einsum('oi,bi->bo', (self.weight, x)) + self.bias
        return self.y


#custom implementation of convolutional layer
class HebbianConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.filter = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(out_channels))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
    def forward(self, x):
        self.x = x
        self.y = F.conv2d(x, self.filter, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return self.y
    
    def hebb_rule(self, rate=0.01):
        xpatches = F.unfold(self.x, self.kernel_size, self.stride, self.padding, self.dilation)
        ypatches = F.unfold(self.y, 1, 1, 0, 1)

        #hebbian learning rule: dw_ij = (x_i) * (y_j)
        dw = torch.einsum('bip,bop->io', (xpatches, ypatches)) 
        if self.bias is not None:
            db = ypatches.sum(dim=1)
            
        dw = dw / xpatches.shape[0]
        raise NotImplementedError("")
        
        
        

def main():
    train = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist', train=True, download=True, transform=torchvision.transforms.ToTensor()), batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(range(640)))
    test = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist', train=False, download=True, transform=torchvision.transforms.ToTensor()), batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(range(640)))

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['KICKBACK_DEVICE'] = str(device)
    
    print("Using device: ", device)
    
    net = KickbackNet(in_dim=784, out_dim=10, hidden_dim=100, num_layers=3)
    net = net.to(device)
    
    epochs = 20
    optimizer = torch.optim.Adam(net.head.parameters(), lr=0.001)

    for epoch in range(epochs):
        print("Beginning training, epoch: ", epoch+1, "/", epochs, "")
        for batch in train:
            x, y = batch
            x = x.view(x.shape[0], -1).to(device)
            y = torch.nn.functional.one_hot(y, num_classes=10).float().to(device)
            z = net.forward(x)
            net.oja_rule(rate=0.01)
            loss = torch.nn.functional.binary_cross_entropy(z, y, reduction='none')
            #reduce over classes
            loss = loss.mean(dim=1)
            #net.full_kickback_rule(loss)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
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
            test_acc += (z.argmax(dim=1) == y.argmax(dim=1)).float().mean().item()
        test_loss /= len(test)
        test_acc /= len(test)
        print("Test loss: ", test_loss, "Test accuracy: ", test_acc)
        print("")
        
    #re-initialize weights
    net = KickbackNet(in_dim=784, out_dim=10, hidden_dim=100, num_layers=2)
    net = net.to(device)
    
    #train end-to-end
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(epochs):
        print("Beginning training, epoch: ", epoch+1, "/", epochs, "")
        for batch in train:
            x, y = batch
            x = x.view(x.shape[0], -1).to(device)
            y = F.one_hot(y, num_classes=10).float().to(device)
            z = net(x)
            loss = F.binary_cross_entropy(z, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        #get test loss and accuracy
        test_loss = 0
        test_acc = 0
        for batch in test:
            x, y = batch
            x = x.view(x.shape[0], -1).to(device)
            y = torch.nn.functional.one_hot(y, num_classes=10).float().to(device)
            z = net.forward(x, endtoend=True)
            loss = torch.nn.functional.binary_cross_entropy(z, y)
            test_loss += loss.item()
            test_acc += (z.argmax(dim=1) == y.argmax(dim=1)).float().mean().item()
        test_loss /= len(test)
        test_acc /= len(test)
        print("Test loss: ", test_loss, "Test accuracy: ", test_acc)
        print("")
        

if __name__ == '__main__':
    main()
    