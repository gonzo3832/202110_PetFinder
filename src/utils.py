import os
from pandas.core.frame import DataFrame
import torch
import random
import numpy as np
import logging
import pandas as pd

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



def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    logger.info('hello')

    set_seed()


if __name__ == '__main__':
    main()
#    df = pd.read_csv('input/petfinder-pawpularity-score/train.csv')
#    df = get_debug_df(df)
#    print(df.head(50))