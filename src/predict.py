import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import gc

def predict(model, device, test_loader):
    #model.to(device)    
    model.eval()
    pred_list = []
    pred_temp = []
    for batch_idx,(data)in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            temp = [t.numpy() for t in torch.sigmoid(model(data).cpu())]
            #pred = torch.sigmoid(model(data))
        #pred = pred.detach().cpu().numpy()    
        pred_list.append(temp[0])
     
        

# ToDo 行列演算の形で書く。for loopは遅いしダサい
    for row in range(len(pred_list)):
        index_at_max = np.argmax(pred_list[row])
        probability = np.max(pred_list[row])
        if index_at_max == 0:
            pred_temp.append( 1 - probability)
        else:
            pred_temp.append( probability)



    
    gc.collect()
    torch.cuda.empty_cache()
    del data

    return np.squeeze(pred_temp)