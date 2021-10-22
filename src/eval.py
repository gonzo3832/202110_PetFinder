import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import gc

def get_epoch_loss_score(model, device, valid_loader, loss_func):
    model.eval()
    epoch_valid_loss = 0

    y_pred_list = []
    y_true_list = []
    for batch_idx, (data, target) in enumerate(valid_loader):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        loss = loss_func(output.view_as(target), target)
        epoch_valid_loss += loss.item()
        
        output = nn.Sigmoid()(output)
        _y_pred = output.detach().cpu().numpy()

#        my_round_int =lambda x : int(( x * 2 + 1)//2)
#       _y_pred = map(my_round_int,_y_pred.squeeze(1))
        y_pred_list.append(_y_pred)

        _y_true = target.detach().cpu().numpy()
        y_true_list.append(_y_true)
    
    loss = epoch_valid_loss / len(valid_loader)
    
    y_pred = np.concatenate(y_pred_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)
    
#    f_score = f1_score(y_true, y_pred, average='macro')
    #auc_score = roc_auc_score(y_true, y_pred)
    
    gc.collect()
    torch.cuda.empty_cache()
    del data,target

    return loss