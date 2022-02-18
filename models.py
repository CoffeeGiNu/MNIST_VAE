import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim) -> None:
        super(Encoder, self).__init__()
        self.eps = np.spacing(1)
        self.x_dim = x_dim
        self.z_dim = z_dim
        
        self.fc1 = nn.Linear(x_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.mean = nn.Linear(256, z_dim)
        self.log_var = nn.Linear(256, z_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_var = self.log_var(x)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim) -> None:
        super(Decoder, self).__init__()
        self.eps = np.spacing(1)
        self.x_dim = x_dim
        self.z_dim = z_dim
        
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.drop = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(512, 784)
    
    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = self.drop(z)
        dec_y = torch.sigmoid(self.fc3(z))

        return dec_y


class VariationalAutoEncoder(nn.Module):
    def __init__(self, x_dim, z_dim, device) -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.eps = np.spacing(1)
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.encoder = Encoder(x_dim, z_dim)
        self.decoder = Decoder(x_dim, z_dim)
    
    def pseudo_sample(self, mean, log_var):
        rand = torch.randn(mean.shape, device=self.device)
        z = mean + rand * torch.exp(1/2 * log_var)

        return z
    
    def forward(self, x):
        x = x.view(-1, self.x_dim)
        mean, log_var = self.encoder(x)
        z = self.pseudo_sample(mean, log_var)
        y = self.decoder(z)
        KLD = 1/2 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))
        rc = torch.sum(x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps))

        return [KLD, rc], z, y