import os
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from datasets import load_tfds, pre_train_preprocessing


num_epochs = 64
optimizer = optim.Adam()


if __name__ == "__main__":
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    
    dataset_train, dataset_valid, dataset_test = load_tfds("mnist", 
        batch_size=batch_size, preprocess_fn=pre_train_preprocessing, seed=seed)
