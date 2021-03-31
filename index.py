import torch
from torch import nn
from torchsummary import summary

from models import Resnet34

if __name__ == '__main__':
    model = Resnet34()
    model.eval()
    state_dict = torch.load('checkpoints/resnet34_state.ckpt')
    model.load_state_dict(state_dict)
    summary(model, (3, 32, 32))
