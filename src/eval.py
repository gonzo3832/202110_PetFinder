import numpy as np
import torch
import gc

def valid(model, device, valid_loader, loss_func):
    model.eval()
    loss = 0
    pred_list = []
    true_list = []

    for data, target in valid_loader:
        data = [data.to(device) for data in data ]
        target =target.to(device)
        with torch.no_grad():
            output = model(*data)
        output = output.view_as(target)
        batch_loss = loss_func(output, target)

        loss += batch_loss.item()
        
        _pred = output.detach().cpu().numpy()
        pred_list.append(_pred)

        _true = target.detach().cpu().numpy()
        true_list.append(_true)
    
    loss = loss / len(valid_loader)
    
    y_pred = np.concatenate(pred_list, axis=0)
    y_true = np.concatenate(true_list, axis=0)
    
    gc.collect()
    torch.cuda.empty_cache()
    del data, target

    return loss,y_pred,y_true