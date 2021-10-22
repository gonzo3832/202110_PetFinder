import logging
import torch
from torch.cuda.amp import GradScaler, autocast
import gc


def train(model, device, train_loader, optimizer, scheduler, loss_func,use_amp):
    scaler = GradScaler(enabled=use_amp)
    model.train()
    epoch_train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            output = model(data)
            loss = loss_func(output.view_as(target), target)
        
        scaler.scale(loss).backward
        scaler.step(optimizer)
        scaler.update()


        epoch_train_loss += loss.item()
    scheduler.step()
    loss = epoch_train_loss / len(train_loader)
    
    gc.collect()
    torch.cuda.empty_cache()
    del data,target

    return loss
