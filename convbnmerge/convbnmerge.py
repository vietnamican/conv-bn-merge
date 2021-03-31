import torch
from torch import nn

def get_parent_of_module(model, path):
    trace = path.split('.')
    first_name = trace[-1]
    temp = model
    for node in trace[:-1]:
        if node.isnumeric():
            temp = temp[int(node)]
        else:
            temp = getattr(temp, node)
    return temp, first_name


def set_module_to_a_model(model, path, module):
    parent, name = get_parent_of_module(model, path)
    setattr(parent, name, module)


def get_equipvalent_conv(conv):
    attrs = ['in_channels', 'out_channels', 'kernel_size',
             'stride', 'padding', 'dilation', 'groups', 'padding_mode']
    bias = True
    args = list(map(lambda x: getattr(conv, x), attrs))
    args[7:7] = [bias]
    new_conv = nn.Conv2d(*args)
    return new_conv


def get_equipvalent_bn(bn):
    return nn.Identity()


def get_equipvalent_conv_bn(conv, bn):
    kernel = conv.weight
    bias = conv.bias
    if bias is None:
        bias = 0
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    return kernel * (gamma / (running_var + eps).sqrt()).reshape(-1, 1, 1, 1), beta + gamma / (running_var + eps).sqrt() * (bias - running_mean)


def merge_conv_bn(conv, bn):
    weight, bias = get_equipvalent_conv_bn(conv, bn)
    new_conv = get_equipvalent_conv(conv)
    with torch.no_grad():
        new_conv.weight.copy_(weight)
        new_conv.bias.copy_(bias)

    new_bn = get_equipvalent_bn(bn)
    return new_conv, new_bn


def merge(model):
    names = [name for name, _ in model.named_modules()]
    modules = [module for module in model.modules()]
    for (current_name, current_module), (next_name, next_module) in zip(zip(names[:-1], modules[:-1]), zip(names[1:], modules[1:])):
        if isinstance(current_module, nn.Conv2d) and isinstance(next_module, nn.BatchNorm2d):
            print('Merging conv {} vs bn {}'.format(current_name, next_name))
            conv, bn = merge_conv_bn(current_module, next_module)
            set_module_to_a_model(model, current_name, conv)
            set_module_to_a_model(model, next_name, bn)