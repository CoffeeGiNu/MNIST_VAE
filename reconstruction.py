import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import fix_seed
from models import VariationalAutoEncoder
from datasets import load_tfds


if __name__ == "__main__":
    log_dir = "./logs"
    seed = 42
    fix_seed(seed)
    x_dim = 28 * 28
    z_dim = 10
    batch_size = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, dataset_test = load_tfds("mnist", batch_size=batch_size, seed=seed, preprocess_fn=None)
    
    model = VariationalAutoEncoder(x_dim, z_dim, device)
    model.load_state_dict(torch.load("./models/checkpoint_z3.pth"))
    model.eval()

    for inputs in dataset_test:
        inputs = inputs['image'] / 255
        inputs = torch.from_numpy(np.array(inputs))
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        # for a in axes:
        #     a.set_xticks([])
        #     a.set_yticks([])

        _, _, y = model(inputs)
        image_x = inputs.cpu().detach().numpy().reshape(-1, 28, 28)
        image_y = y.cpu().detach().numpy().reshape(-1, 28, 28)
        for j in range(batch_size):
            axes[0][j].imshow(image_x[j], "gray")
            axes[0][j].set_xticks([])
            axes[0][j].set_yticks([])
            axes[1][j].imshow(image_y[j], "gray")
            axes[1][j].set_xticks([])
            axes[1][j].set_yticks([])
        fig.savefig(f"./figure/reconstracttion_z{z_dim}.png")
        plt.close()
        break