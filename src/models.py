import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm1d
import timm
from torchsummary import summary


def get_model(config):
    model_config = config["model"]
    model_name = model_config["name"]
    model_params = model_config["params"]
    model = eval(model_name)(model_params)
    return model


class EffNetV2_b0(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.model = timm.create_model('tf_efficientnetv2_b0', pretrained=True, in_chans=3)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, params['num_classes'])

    def forward(self, x):
        out = self.model(x)
        return out.squeeze()


class EffV2_plus_table(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.model = timm.create_model('tf_efficientnetv2_b0', pretrained=True, in_chans=3)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 10)
        self.linear1 = nn.Linear(self.model.classifier.out_features + 12, 20)
        self.norm1 = BatchNorm1d(self.linear1.out_features)
        self.linear2 = nn.Linear(20, 20)
        self.norm2 = BatchNorm1d(self.linear2.out_features)
        self.fc = nn.Linear(20, params["num_classes"])

    def forward(self, x, table):
        out = self.model(x)
        out = torch.cat([out, table],1) # NOTE:catは次元変わらない。stackは変わる
        out = self.linear1(out)
        out = Swish()(self.norm1(out))
        out = self.linear2(out)
        out = Swish()(self.norm2(out))
        out = self.fc(out)
        return out.squeeze()


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x*torch.sigmoid(x)


if __name__ == '__main__':

    # ------- debug code -------
    import pandas as pd
    from datasets import image_plus_table
    import torch.utils.data as data

    tensorA = torch.tensor([[0, 1, 2]])
    tensorB = torch.tensor([[3, 4, 5]])
    stacked = torch.stack([tensorA, tensorB], dim=1)
    print(stacked.size())
    params = {'num_classes': 1}
    model = EffV2_plus_table(params)
#    summary(model,(3,256,256))
    df = pd.read_csv("input/petfinder-pawpularity-score/train.csv")
    print(df.head())
    datadir = "input/petfinder-pawpularity-score/train"
    phase = "train"
    cfg = {"img_size": 224}
    datasets = image_plus_table(df, datadir, phase, cfg)
    dataloader = data.DataLoader(datasets, 3)
    for data, target in dataloader:
        #        x, table = data
        #        x = x.unsqueeze(0)
        #        table = table.unsqueeze(0)
        print(model(*data))
        break
