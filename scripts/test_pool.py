import torch
import torch.nn.functional as Funct
from torch.autograd import Variable
import numpy as np


def test_pool(img_file):
    img = Variable(torch.load(img_file))

    maxpool, idx = Funct.max_pool2d(img, (2, 2), stride=(2, 2), return_indices=True)
    maxunpool = Funct.max_unpool2d(maxpool, idx, kernel_size=(2, 2), stride=(2, 2))
    avgpool = Funct.avg_pool2d(img, (2, 2), stride=(2, 2))

    return [
        np.float32(maxpool.data.numpy()),
        np.float32(maxunpool.data.numpy()),
        np.float32(avgpool.data.numpy())
    ]