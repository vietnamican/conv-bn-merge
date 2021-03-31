from typing import Dict, Iterable, List, Optional, Union
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl


class BaseException(Exception):
    def __init__(
            self,
            parameter,
            types: List):
        message = '{} type must be one of {}'.format(parameter, types)
        super().__init__(message)


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.is_released = False

    def remove_num_batches_tracked(self, state_dict):
        new_state_dict = {}
        for name, p in state_dict.items():
            if not 'num_batches_tracked' in name:
                new_state_dict[name] = p
        return new_state_dict

    def migrate(
            self,
            state_dict: Dict,
            force=False,
            verbose=2
    ):
        '''
        verbose=0: do not print
        verbose=1: print status of migrate: all is migrated or something
        verbose=2: print all of modules had been migrated
        '''
        if verbose == 0:
            def status(i, string):
                pass

            def conclude(is_all_migrated):
                pass
        elif verbose == 1:
            def status(i, string):
                pass

            def conclude(is_all_migrated):
                if is_all_migrated:
                    print("all modules had been migrated")
                else:
                    print("Some modules hadn't been migrated")
        elif verbose == 2:
            def status(i, string):
                print(i, string)

            def conclude(is_all_migrated):
                if is_all_migrated:
                    print("all modules had been migrated")
                else:
                    print("Some modules hadn't been migrated")

        des_state_dict = self.state_dict()
        source_state_dict = state_dict

        des_state_dict = self.remove_num_batches_tracked(des_state_dict)
        source_state_dict = self.remove_num_batches_tracked(source_state_dict)
        is_all_migrated = True

        if not force:
            state_dict_keys = source_state_dict.keys()
            with torch.no_grad():
                for i, (name, p) in enumerate(des_state_dict.items()):
                    if name in state_dict_keys:
                        _p = source_state_dict[name]
                        if p.data.shape == _p.shape:
                            status(i, name)
                            p.copy_(_p)
                        else:
                            is_all_migrated = False
                    else:
                        is_all_migrated = False

        else:
            print('Force migrating...')
            with torch.no_grad():
                for i, ((name, p), (_name, _p)) in enumerate(zip(des_state_dict.items(), source_state_dict.items())):
                    if p.shape == _p.shape:
                        status(i, 'copy to {} from {}'.format(name, _name))
                        p.copy_(_p)
                    else:
                        is_all_migrated = False
        conclude(is_all_migrated)

    @staticmethod
    def remove_prefix_state_dict(
            state_dict: Dict,
            prefix: Union[str, int]
    ):
        result_state_dict = {}
        if isinstance(prefix, int):
            # TODO
            return state_dict
        elif isinstance(prefix, str):
            len_prefix_remove = len(prefix) + 1
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    result_state_dict[key[len_prefix_remove:]
                                      ] = state_dict[key]
                else:
                    result_state_dict[key] = state_dict[key]
            return result_state_dict
        else:
            raise BaseException('prefix', [str, int])

    @staticmethod
    def filter_state_dict_with_prefix(
        state_dict: Dict,
        prefix: str,
        is_remove_prefix=False
    ):
        if not isinstance(prefix, str):
            raise BaseException('prefix', [str])
        new_state_dict = {}
        if is_remove_prefix:
            prefix_length = len(prefix)
            for name, p in state_dict.items():
                if name.startswith(prefix):
                    new_state_dict[name[prefix_length+1:]] = p
        else:
            for name, p in state_dict.items():
                if name.startswith(prefix):
                    new_state_dict[name] = p
        return new_state_dict

    @staticmethod
    def filter_state_dict_except_prefix(
        state_dict: Dict,
        prefix: str,
    ):
        if not isinstance(prefix, str):
            raise BaseException('prefix', [str])
        new_state_dict = {}
        for name, p in state_dict.items():
            if not name.startswith(prefix):
                new_state_dict[name] = p
        return new_state_dict

    def freeze_except_prefix(self, prefix):
        for name, p in self.named_parameters():
            if not name.startswith(prefix):
                p.requires_grad = False

    def freeze_with_prefix(self, prefix):
        for name, p in self.named_parameters():
            if name.startswith(prefix):
                p.requires_grad = False

    def defrost_except_prefix(self, prefix):
        for name, p in self.named_parameters():
            if not name.startswith(prefix):
                p.requires_grad = True

    def defrost_with_prefix(self, prefix):
        for name, p in self.named_parameters():
            if name.startswith(prefix):
                p.requires_grad = True


class BaseSequential(nn.Sequential, Base):
    pass


class ConvBatchNormRelu(Base):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if 'with_relu' not in kwargs:
            self.with_relu = True
        else:
            self.with_relu = kwargs['with_relu']
            kwargs.pop('with_relu', None)
        if 'with_bn' not in kwargs:
            self.with_bn = True
        else:
            self.with_bn = kwargs['with_bn']
            kwargs.pop('with_bn', None)

        self.args = args
        self.kwargs = kwargs

        self.cbr = nn.Sequential()
        self.cbr.add_module('conv', nn.Conv2d(*args, **kwargs))
        if self.with_bn:
            outplanes = args[1]
            self.cbr.add_module('bn', nn.BatchNorm2d(int(outplanes)))
        if self.with_relu:
            self.cbr.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.cbr(x)


class BasicBlock(Base):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=False,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ConvBatchNormRelu(
            inplanes, planes, kernel_size=3, padding=1, bias=False, stride=stride, with_relu=False)
        self.identity_layer = ConvBatchNormRelu(
            inplanes, planes, kernel_size=1, padding=0, bias=False, stride=stride, with_relu=False)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        if inplanes == planes and stride == 1:
            self.skip_layer = nn.BatchNorm2d(num_features=inplanes)
        else:
            self.skip_layer = None
        if self.skip_layer is not None:

            def _forward(self, x):
                conv3 = self.conv1(x)
                identity = self.identity_layer(x)
                skip = self.skip_layer(x)
                return self.relu(conv3 + identity + skip)
        else:

            def _forward(self, x):
                conv3 = self.conv1(x)
                identity = self.identity_layer(x)
                return self.relu(conv3 + identity)
        self._forward = partial(_forward, self)

    def forward(self, x):
        return self._forward(x)
