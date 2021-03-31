from torch import nn

def get_parent_of_module(model, path):
    pass


def set_module_to_a_model(model, path, module):
    pass


def merge(model):
    names = [name for name, _ in model.named_modules()]
    modules = [module for module in model.modules()]
    for (current_name, current_module), (next_name, next_module) in zip(zip(names[:-1], modules[:-1]), zip(names[1:], modules[1:])):
        if isinstance(current_module, nn.Conv2d) and isinstance(next_module, nn.BatchNorm2d):
            print('Merging conv {} vs bn {}'.format(current_name, next_name))
