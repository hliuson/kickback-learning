import torch

class Triangle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        x = x - torch.mean(x, dim=1, keepdim=True)
        return F.relu(x)