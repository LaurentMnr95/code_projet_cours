import numpy as np
from torch import nn
import torch
import os
import numpy as np
import uuid
from pathlib import Path


def get_shared_folder() -> Path:
    if Path("/checkpoint/").is_dir():
        return Path("/checkpoint/laurentmeunier/cifar")
    raise RuntimeError("No shared folder available")


def get_init_file() -> Path:
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.normalize = Normalize(mean, std)

    def forward(self, x):
        return self.model(self.normalize(x))


class MixedModel(nn.Module):
    def __init__(self, models, lbda):
        super(MixedModel, self).__init__()
        self.models = torch.nn.ModuleList(models)
        self.lbda = lbda

    def forward(self, x):
        ii = np.random.choice(np.arange(len(self.lbda)), p=self.lbda)
        return self.models[ii](x)