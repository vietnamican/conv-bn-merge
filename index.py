from time import time

import torch
from torch import nn
from torchsummary import summary

from models import Resnet34
from merge import merge

if __name__ == '__main__':
    model = Resnet34()
    # model.eval()
    # state_dict = torch.load('checkpoints/resnet34_state.ckpt')
    # model.load_state_dict(state_dict)
    # summary(model, (3, 32, 32), depth=6)
    x = torch.Tensor(1, 3, 32, 32)
    start = time()
    for i in range(200):
        model(x)
    stop = time()
    print(stop - start)
    merge(model)
    start = time()
    for i in range(200):
        model(x)
    stop = time()
    print(stop - start)
    # summary(model, (3, 32, 32), depth=6)
