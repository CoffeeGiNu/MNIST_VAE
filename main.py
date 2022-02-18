import os
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from utils import fix_seed
from train import epoch_loop
from callbacks import EarlyStopping
from models import VariationalAutoEncoder
from datasets import load_tfds, pre_train_preprocessing


if __name__ == "__main__":
    seed = 42
    fix_seed(seed)
    x_dim = 28 * 28
    z_dim = 10
    batch_size = 128
    num_epochs = 1000
    learning_rate = 0.001
    loss_fn = lambda lower_bound: -sum(lower_bound)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs("./logs") if not os.path.exists("./logs") else None
    writer = SummaryWriter(log_dir="./logs")
    
    dataset_train, dataset_valid, dataset_test = load_tfds("mnist", 
        batch_size=batch_size, preprocess_fn=pre_train_preprocessing, seed=seed)
    
    model = VariationalAutoEncoder(x_dim, z_dim, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    earlystopping = EarlyStopping()

    for e in range(1, num_epochs+1):
        model = epoch_loop(model, dataset_train, optimizer, loss_fn, device, e, num_epochs, batch_size, is_train=True, writer=writer)
        model = epoch_loop(model, dataset_valid, optimizer, loss_fn, device, e, num_epochs, batch_size, is_train=False, earlystopping=earlystopping, writer=writer)
        if earlystopping.early_stop:
            break
    writer.close()