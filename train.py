import sys
from tqdm.auto import tqdm

import torch
import torchvision
import numpy as np


def step(model, inputs, optimizer, criterion, device, is_train=True):
    model = model.to(device)
    inputs = torch.from_numpy(np.array(inputs)).to(device)
    optimizer.zero_grad()
    with torch.set_grad_enabled(is_train):
        lower_bound, z, y = model(inputs)
        loss = criterion(lower_bound)
        if is_train:
            loss.backward()
            optimizer.step()
        
    return model, loss, lower_bound, z, y


def epoch_loop(model, data_set, optimizer, criterion, device, epoch, num_epochs, batch_size, earlystopping=None, is_train=True, writer=None):
    with tqdm(
        total=len(data_set),
        bar_format=None if 'ipykernel' in sys.modules else '{l_bar}{bar:15}{r_bar}{bar:-10b}',
        ncols=None if 'ipykernel' in sys.modules else 108,
        unit='batch',
        leave=True
    ) as pbar:
        total = loss_sum = accuracy_sum = 0
        pbar.set_description(
            f"Epoch[{epoch}/{num_epochs}]({'train' if is_train else 'valid'})")
        for data in data_set:
            inputs = data['image'] / 255
            model, loss, lower_bound, z, y = step(
                model, inputs, optimizer, criterion, device, is_train=is_train)
            # if writer:
            #     writer.add_scalar("Loss_train/KLD", -lower_bound[0].cpu().detach().numpy(), epoch + total)
            #     writer.add_scalar("Loss_train/Reconst", -lower_bound[1].cpu().detach().numpy(), epoch + total)
            total += batch_size
            loss_sum += loss * batch_size
            running_loss = loss_sum.item() / total
            # accuracy_sum += (torch.argmax(preds, axis=1).detach().cpu().numpy() == labels).sum()
            # running_accuracy = accuracy_sum.item() / total
            pbar.set_postfix(
                {"loss":round(running_loss, 3), 
                #  "accuracy":round(running_accuracy, 3)
                }
            )
            writer.step()
            pbar.update(1)
        if earlystopping:
            earlystopping((running_loss), model)
    
    return model