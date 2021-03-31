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
    # outplanes, inplanes, kernel_h, kernel_w = conv.weight.shape
    # padding = (kernel_h // 2, kernel_w // 2)
    # new_module = nn.Conv2d(inplanes, outplanes, kernel_size=(kernel_h, kernel_w), padding=padding, stride=1)
    return conv

def get_equipvalent_bn(bn):
    return nn.Identity()

def merge(model):
    names = [name for name, _ in model.named_modules()]
    modules = [module for module in model.modules()]
    for (current_name, current_module), (next_name, next_module) in zip(zip(names[:-1], modules[:-1]), zip(names[1:], modules[1:])):
        if isinstance(current_module, nn.Conv2d) and isinstance(next_module, nn.BatchNorm2d):
            print('Merging conv {} vs bn {}'.format(current_name, next_name))
            new_conv = get_equipvalent_conv(current_module)
            set_module_to_a_model(model, current_name, new_conv)

            new_bn = get_equipvalent_bn(next_module)
            set_module_to_a_model(model, next_name, new_bn)
            
