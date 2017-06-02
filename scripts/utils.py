import torch
import numpy as np


def save_tensor(n, k, h, w, filename):
    tensor = torch.zeros(n, k, h, w).uniform_(0, .5)
    torch.save(tensor, filename)


def load_tensor(filename):
    tensor = torch.load(filename)
    return np.float32(tensor.cpu().numpy())
