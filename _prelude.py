from dataclasses import dataclass

import numpy as np 
import torch 
import torch.nn as nn 

from typing import Optional

from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import save_image

from PIL import Image
import png 

import unittest
import os

def open_image(path: str) -> torch.Tensor:
    return transforms.ToTensor()(Image.open(path)).unsqueeze(0)[:, :3, :, :]
