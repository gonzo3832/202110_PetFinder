import logging
import torch
from torch.cuda.amp import GradScaler, autocast
import gc
import criterion
from fastprogress import progress_bar


def train(model, device, train_loader, optimizer, scheduler, loss_func,use_amp):
    scaler = GradScaler(enabled=use_amp)
    model.train()
    epoch_train_loss = 0
    epoch_train_loss_rmse = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            output = model(data)
            loss = loss_func(output.view_as(target), target)
            rmse_loss = RMSELoss(output.view_as(target), target)*100
            
        
        scaler.scale(loss).backward
        scaler.step(optimizer)
        scaler.update()


        epoch_train_loss += loss.item()
        epoch_train_loss_rmse += rmse_loss.item()

    scheduler.step()
    loss = epoch_train_loss / len(train_loader)
    loss_rmse = epoch_train_loss_rmse/len(train_loader)

    gc.collect()
    torch.cuda.empty_cache()
    del data,target

    return loss,loss_rmse


def train_fastprogress(model, device, train_loader, optimizer, scheduler, loss_func,mb):
    model.train()

    epoch_train_loss = 0
    epoch_train_loss_rmse = 0
    RMSE = criterion.RMSELoss()

    loader = train_loader.__iter__()


    for _ in progress_bar(range(len(loader)),parent=mb): 
        
        data, target = loader.next()
        data, target = data.to(device), target.to(device)
        output = model(data)
        print(type(output))
        print(type(target))
        loss = loss_func(output, target)
        rmse_loss = RMSE(torch.sigmoid(output), target)*100
            
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        epoch_train_loss_rmse += rmse_loss.item()

    scheduler.step()
    loss = epoch_train_loss / len(train_loader)
    loss_rmse = epoch_train_loss_rmse/len(train_loader)

    gc.collect()
    torch.cuda.empty_cache()
    del data,target

    return loss,loss_rmse

def train_simple(model, device, train_loader, optimizer, scheduler, loss_func,mb):
    model.train()
    model.to(device)
    loss = 0

    loader = train_loader.__iter__()

    for _ in progress_bar( range(len(loader)),parent=mb):
        data, target = loader.next()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.view_as(target)
        
        batch_loss = loss_func(output, target)
            
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()

    scheduler.step()
    loss = loss / len(train_loader)

    gc.collect()
    torch.cuda.empty_cache()
    del data, target

    return loss
