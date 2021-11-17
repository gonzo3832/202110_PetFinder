# -*- coding: utf-8 -*-
from typing import Awaitable
import warnings

warnings.filterwarnings('ignore')
import os
import gc
import hydra
import torch
import logging
import subprocess
from omegaconf import DictConfig
import pandas as pd
import numpy as np


from src import utils
from src import configuration as C
from src import models
from src.early_stopping import EarlyStopping
from src.train import train_fastprogress
from src.eval import get_epoch_loss_score
import src.result_handler as rh
import dill
import yaml
from fastprogress import master_bar, progress_bar

cmd = "git rev-parse --short HEAD"
hash_ = subprocess.check_output(cmd.split()).strip().decode('utf-8') #subprocessで上のコマンドを実行したのち、結果を格納
logger = logging.getLogger(__name__)
config_path = '.'
config_name = 'run_config_simple.yaml'

@hydra.main(config_path=config_path, config_name=config_name)
def run (cfg: DictConfig) -> None:

    #time logging start
    logger.info(f'git hash is: {hash_}')
    df, datadir = C.get_metadata(cfg)

    if cfg['globals']['debug']:
        logger.info('::: set debug mode :::')
        cfg = utils.get_debug_config(cfg)
        df = utils.get_debug_df(df)

    global_params = cfg["globals"]
    utils.set_seed(global_params['seed'])
    device = C.get_device()
    splitter = C.get_split(cfg)
    comment = global_params['comment']
    
    logger.info(f'meta_df: {df.shape}')
    output_dir = os.getcwd()
    output_dir_ignore = output_dir.replace('/outputs/', '/model_weight/')
    if not os.path.exists(output_dir_ignore):
            os.makedirs(output_dir_ignore)

    model_paths = {}

    n_epoch = cfg['globals']['num_epochs']
    x_bounds = [0, n_epoch]
    y_bounds = [0,1]


    for fold_i, (trn_idx, val_idx) in enumerate(
        splitter.split(df, y=df['Pawpularity'])
        ):
       
        logger.info(f'Fold{fold_i}')
    
        # ------- define datasets and dataloader ----
        trn_df = df.loc[trn_idx, :].reset_index(drop=True)
        trn_df = utils.min_max_normalize(trn_df,'Pawpularity',0,100)
        val_df = df.loc[val_idx, :].reset_index(drop=True)
        val_df = utils.min_max_normalize(val_df,'Pawpularity',0,100)
        
        logger.info(f'trn_df: {trn_df.shape}')
        logger.info(f'val_df: {val_df.shape}')

        logger.info(f"dataset: { cfg['dataset']['name']}")
        train_loader = C.get_loader(trn_df, datadir, cfg, 'train')
        valid_loader = C.get_loader(val_df, datadir, cfg, 'valid')

        # ------ define actor ----------
        model = models.get_model(cfg).to(device)
        criterion = C.get_criterion(cfg).to(device)
        optimizer = C.get_optimizer(model, cfg)
        scheduler = C.get_scheduler(optimizer, cfg)
    
        # ------ define tracker -------
        losses_train = []
        losses_valid = []
        losses_rmse_train = []
        losses_rmse_valid = []
        epochs = []
        best_loss = 0
        best_loss_rmse = 0

        # ------- model saving condition -----
        model_path = f'{output_dir}/{model.__class__.__name__}.pth'
        model_paths[fold_i] = model_path
        early_stopping = EarlyStopping(**cfg['early_stopping'],verbose=True, path=model_path,device=device)
        
        
        
        mb = master_bar(range(1,n_epoch))
        for epoch in mb:
            logger.info(f'::: epoch: {epoch}/{n_epoch} :::')            
            
            loss_train ,loss_rmse_train= train_fastprogress(
                model, device, train_loader, optimizer,
                scheduler, criterion,mb
            )    
            
            loss_valid ,loss_rmse_valid = get_epoch_loss_score(
                model, device, valid_loader, criterion
            )

            logger.info(f'loss_train: {loss_train:.4f}, loss_valid: {loss_valid:.4f},loss_rmse_train: {loss_rmse_train:.4f}, loss_rmse_valid: {loss_rmse_valid:.4f}')
            
            epochs.append(epoch)
            losses_train.append(loss_train)
            losses_valid.append(loss_valid)
            losses_rmse_train.append(loss_rmse_train)
            losses_rmse_valid.append(loss_rmse_valid)

            is_update = early_stopping(loss_valid, model, debug=False)
            if is_update:
                best_loss = loss_valid
                best_loss_rmse = loss_rmse_valid
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

            t = np.arange(epoch)
            graphs = [[t,losses_train],[t,losses_valid],[t,losses_rmse_train],[t,losses_rmse_valid]]
            mb.update_graph (graphs,x_bounds,y_bounds)
            mb.write('EPOCH: {0:02d}, Train loss: {1:10.5f}, Valid loss: {2:10.5f}, train loss(RMSE): {3:10.5f}, valid loss(RMSE): {4:10.5f}'.format(
            epoch, loss_train, loss_valid,loss_rmse_train,loss_rmse_valid))

        '''
        rh.save_loss_figure(
            fold_i,
            epochs, losses_train,
            losses_valid, 
            losses_rmse_train,
            losses_rmse_valid,
            output_dir
        )
        rh.save_result_csv(
            fold_i,
            global_params['debug'], 
            f'model.__class__.__name__',
            cfg['loss']['name'], 
            best_loss, 
            best_loss_rmse,
            comment,
            output_dir
        )
        '''
        logger.info(f'best_loss: {best_loss:.6f}')
    
    del train_loader
    del valid_loader
    del model
    del optimizer
    del scheduler
    gc.collect()
    torch.cuda.empty_cache()
    # ToDo : separate train mode and predict mode
    # dfで行くか、最後に足し算する感じで
    # fold ごとに行を作成して出力
    
    logger.info('::: success :::\n\n\n')

if __name__ == "__main__":
    yaml_path = os.path.join(config_path,config_name)
    
    with open(yaml_path,'r+') as f:
        cfg = yaml.safe_load(f)
        if cfg['globals']['comment'] == None:
            comment = input('commentを入力してください：')
            cfg['globals']['comment'] = comment
        f.seek(0)
        yaml.safe_dump(cfg, f,sort_keys=False)

    run()

    cfg['globals']['comment'] = None
    with open(yaml_path,'w') as f:
        yaml.safe_dump(cfg,f,sort_keys=False)