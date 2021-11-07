import os
from pandas.core.frame import DataFrame
import torch
import random
import numpy as np
import logging
import pandas as pd
import yaml
from collections import OrderedDict

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = True  # type: ignore

def get_debug_config(config):
    config['globals']['num_epochs'] = 1
    config['split']['params']['n_splits'] = 2
    return config


def get_debug_df(df):
    df = df.groupby("Pawpularity").head(5).sort_values("Pawpularity").reset_index()
    df = df[df['Pawpularity']%10 == 0]
    df = df.reset_index()
    return df


def ensemble(df, n_ignore_columns: int, name_ens_column: str)-> pd.DataFrame:
    n_ensemble = len(df.columns)-n_ignore_columns
    df[name_ens_column] = df[name_ens_column].astype(np.float64)
    for index,row in df.iterrows():
        preds = row[n_ignore_columns:]
        vote = preds >= 0.5
        if vote.sum() == n_ensemble/2:
            if preds[preds >= 0.5].mean() > 1 - preds[preds < 0.5].mean():
                df.at[index, name_ens_column] = preds.max()
            else:
                df.at[index, name_ens_column] = preds.min()
        elif vote.sum() > n_ensemble/2:
            df.at[index, name_ens_column] = preds.max()
        else:
            df.at[index, name_ens_column] = preds.min()
    
    return df


def min_max_normalize(df,target_column,min,max):
    df[target_column] = (df[target_column]-min)/(max-min)
    return df


def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))



def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    logger.info('hello')

    set_seed()


def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

def ordered_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class OrderedDumper(Dumper):
        pass
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)

# usage example:



if __name__ == '__main__':
    main()
    df = pd.read_csv('input/petfinder-pawpularity-score/train.csv')
    df = min_max_normalize(df,'Pawpularity',0,100)
    print(df.head())
#    df = get_debug_df(df)
#    print(df.head(50))


