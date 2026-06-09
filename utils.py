import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_optim(model):
    return Adam(model.parameters(), lr=0.001)

def get_criterion():
    return CrossEntropyLoss()