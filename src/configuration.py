import os
import sys
import hydra
import pandas as pd
import sklearn.model_selection as sms

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

sys.path.insert(0, f'{os.getcwd()}/src')
import datasets
import criterion
'''
'''

def get_device(device: str):
    return torch.device(device)

def get_split(config: dict):
    split_config = config["split"]
    name = split_config["name"]

    return sms.__getattribute__(name)(**split_config["params"])

def get_metadata(config: dict):
    ori_path = hydra.utils.get_original_cwd()
    # hydraが走ってるとこのpath取得
    data_config = config["data"]
    path_train_csv = f'{ori_path}/{data_config["train_df_path"]}'
    path_data = f'{ori_path}/{data_config["train_data_dir"]}'

    train = pd.read_csv(path_train_csv)
    return train, path_data

def get_metadata_test(config: dict):
    ori_path = hydra.utils.get_original_cwd()
    # hydraが走ってるとこのpath取得
    data_config = config["data"]
    path_test_csv = f'{ori_path}/{data_config["test_df_path"]}'
    path_data = f'{ori_path}/{data_config["test_data_dir"]}'

    test = pd.read_csv(path_test_csv)
    return test, path_data

def get_loader(df: pd.DataFrame,
               datadir,
               config: dict,
               phase: str,
               ):
    dataset_config = config["dataset"]
    name = dataset_config['name']
    loader_config = config["loader"][phase]

    dataset = datasets.__getattribute__(name)(
            df,
            datadir=datadir,
            phase=phase,
            config=dataset_config['params'],
            )
    loader = data.DataLoader(dataset, **loader_config)
    return loader

def get_criterion(config: dict):
    loss_config = config["loss"]
    loss_name = loss_config["name"]
    loss_params = loss_config["params"]
    if (loss_params is None) or (loss_params == ""):
        loss_params = {}

    if hasattr(nn, loss_name): # torch.nnに同名のloss関数があったら
        criterion_ = nn.__getattribute__(loss_name)(**loss_params)
    else: # ない場合は、自作のcriterion moduleから持ってくる
        criterion_cls = criterion.__getattribute__(loss_name) # getattrで同名クラスを所得して（インスタンス化はまだ）
        if criterion_cls is not None: #  
            criterion_ = criterion_cls(**loss_params) #パラメータ渡してインスタンス化
        else:
            raise NotImplementedError

    return criterion_

def get_optimizer(model: nn.Module, config: dict):
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config.get("name")

    return optim.__getattribute__(optimizer_name)(model.parameters(),
                                                  **optimizer_config["params"])

def get_scheduler(optimizer, config: dict):
    scheduler_config = config["scheduler"]
    scheduler_name = scheduler_config.get("name")

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **scheduler_config["params"])
