import os
import torch
import torch.nn as nn
import sys
import timm

def get_model(config):
    model_config = config["model"]
    model_name = model_config["name"]
    model_params = model_config["params"]

    model = eval(model_name)(model_params)
    # eval関数　：　文字列をpythonのコードとして実行する
    # modelのインスタンス化してることになる
    return model

class EffNetV2_b0(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.model = timm.create_model('tf_efficientnetv2_b0',pretrained=True,in_chans=3)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, params['num_classes'])
        

    def forward(self,x):
        out = self.model(x)
        return out


