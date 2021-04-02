# Convolution Batchnorm Merge

Only one line of code and we can accelerate your model up to 50% faster! 

## Installation
```
$ pip install convbnmerge
```

## Usage
`conv-bn-merge` is <b>ONLY</b> used in inference time!
```python
from convbnmerge import merge

model = ...
"""
training...
"""
merge(model)
```

## Update
* 2021.02.04: also support Conv3d


## How much fast
You usually reach 30++% inferece time reduce. In some cases, the number is more than 50%! 
```python
from time import time

import torch
from torchvision.models.resnet import resnet34

from convbnmerge import merge

if __name__ == '__main__':
    model = resnet34(pretrained=True)
    x = torch.Tensor(2, 3, 32, 32)

    with torch.no_grad():
        start = time()
        for i in range(100):
            model(x)
        stop = time()
        print(stop - start)             # Before merge: about 7.9s
    
    merge(model)

    with torch.no_grad():
        start = time()
        for i in range(100):
            model(x)
        stop = time()
        print(stop - start)             # After merge: about 4.8s
```

## How we do
Coming soon

## Are outputs the same before and after merge?
A small difference caused by round-off error. In almost cases, it doesn't harm the model's result.
```python
import torch
from torchvision.models.resnet import resnet34

from convbnmerge import merge

if __name__ == '__main__':
    model = resnet34(pretrained=True)
    model.eval()
    x = torch.Tensor(1, 3, 32, 32)
    out_old = model(x)
    merge(model)
    out_new = model(x)
    print(((out_old-out_new)**2).sum())         #less than 1e-10 
```

## License
`conv-bn-merge` is MIT-licensed.