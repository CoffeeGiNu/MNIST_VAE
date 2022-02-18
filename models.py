import torch
import numpy as np
import torch.nn as nn
import torch.functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.eps = np.spacing(1)
        self