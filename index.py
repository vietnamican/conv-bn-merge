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
    merge(model)
    # summary(model, (3, 32, 32))
