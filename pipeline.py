# -*- coding: utf-8 -*-
from typing import Awaitable
import warnings

from numpy.lib.shape_base import apply_along_axis
warnings.filterwarnings('ignore')
import os
import gc
import hydra
import torch
import logging
import subprocess
from fastprogress import progress_bar
from omegaconf import DictConfig
from omegaconf import OmegaConf
import pandas as pd
import numpy as np


from src import utils
from src import configuration as C
from src import models
from src.early_stopping import EarlyStopping
from src.train import train
from src.eval import get_epoch_loss_score
from src.predict import predict
import src.result_handler as rh
import dill

cmd = "git rev-parse --short HEAD"
hash_ = subprocess.check_output(cmd.split()).strip().decode('utf-8') #subprocessで上のコマンドを実行したのち、結果を格納
logger = logging.getLogger(__name__)

@hydra.main(config_name="run_config") #カレントディレクトリの左記ファイルを参照する
def run (cfg: DictConfig) -> None:
    # 自分用メモ①：->はアノテーションで、返り値の型を明示している。別になくても問題ない。
    # 自分用メモ②：引数の横の：は想定している変数の型を明示している
    logger.info('='*30) 
    logger.info('::: pipeline start :::')
    logger.info('='*30)
    logger.info(f'git hash is: {hash_}')
    logger.info(f'all params\n{"="*80}\n{OmegaConf.to_yaml(cfg)}\n{"="*80}')
    comment = cfg['globals']['comment']
    assert comment!=None, 'commentを入力してください。(globals.commet=hogehoge)'
    #  assert は条件式がFalseの時に、メッセージを返す。この場合、コメントがない時にメッセージが出る
    df, datadir = C.get_metadata(cfg)
    print(df.head())
    if cfg['globals']['debug']:
        logger.info('::: set debug mode :::')
        cfg = utils.get_debug_config(cfg)
        df = utils.get_debug_df(df)

    global_params = cfg["globals"]
    utils.set_seed(50)
    device = C.get_device(global_params["device"])
    splitter = C.get_split(cfg)
    
    logger.info(f'meta_df: {df.shape}')
    output_dir = os.getcwd()
    output_dir_ignore = output_dir.replace('/data/', '/data_ignore/')
    if not os.path.exists(output_dir_ignore):
            os.makedirs(output_dir_ignore)

    model_paths = {}

    for fold_i, (trn_idx, val_idx) in enumerate(
        splitter.split(df, y=df['Pawpularity'])
        ):
       
        logger.info('='*30)
        logger.info(f'Fold{fold_i}')
        logger.info('='*30)
        trn_df = df.loc[trn_idx, :].reset_index(drop=True)
        trn_df = utils.min_max_normalize(trn_df,'Pawpularity',0,100)
        val_df = df.loc[val_idx, :].reset_index(drop=True)
        val_df = utils.min_max_normalize(val_df,'Pawpularity',0,100)
        
        # splitter 前で正規化すると謎のエラー吐いた（おそらく少数は層化抽出できない）
        logger.info(f'trn_df: {trn_df.shape}')
        logger.info(f'val_df: {val_df.shape}')
        train_loader = C.get_loader(trn_df, datadir, cfg, 'train')
        valid_loader = C.get_loader(val_df, datadir, cfg, 'valid')
    
        model = models.get_model(cfg).to(device)
        criterion = C.get_criterion(cfg).to(device)
        optimizer = C.get_optimizer(model, cfg)
        scheduler = C.get_scheduler(optimizer, cfg)
        losses_train = []
        losses_valid = []
        epochs = []
        best_loss = 0
        model_path = f'{output_dir_ignore}/{model.__class__.__name__}_fold{fold_i}.pth'
        model_paths[fold_i] = model_path
        early_stopping = EarlyStopping(**cfg['early_stopping'],verbose=True, path=model_path,device=device)
        n_epoch = cfg['globals']['num_epochs']
        for epoch in progress_bar(range(1,n_epoch+1)):
            logger.info(f'::: epoch: {epoch}/{n_epoch} :::')
            
            loss_train = train(
                model, device, train_loader, optimizer,
                scheduler, criterion, cfg['globals']['use_amp']
            )    
            
            loss_valid = get_epoch_loss_score(
                model, device, valid_loader, criterion
            )
            logger.info(f'loss_train: {loss_train:.4f}, loss_valid: {loss_valid:.4f}')
            epochs.append(epoch)
            losses_train.append(loss_train)
            losses_valid.append(loss_valid)
            is_update = early_stopping(loss_valid, model, debug=False)
            if is_update:
                best_loss = loss_valid
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
        rh.save_loss_figure(
            fold_i,
            epochs, losses_train,
            losses_valid, output_dir
        )
        rh.save_result_csv(
            fold_i,
            global_params['debug'], 
            f'model.__class__.__name__',
            cfg['loss']['name'], 
            best_loss, 
            comment,
            output_dir
        )
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

    dill.dump(model_paths, open('../../../../../model_paths.pkl', 'wb'))
    model_paths = dill.load(open('../../../../../model_paths.pkl', 'rb'))
    logger.info('::: predict :::')

    test_df, test_data_dir = C.get_metadata_test(cfg)    
    model = models.get_model(cfg).to(device)

    test_loader = C.get_loader(test_df, test_data_dir, cfg, 'test')
    preds_sum = np.zeros(len(test_df))
    n_splits = int(cfg['split']['params']['n_splits'])
    for fold_i in range(n_splits):
        model_path = model_paths[fold_i]
        model.load_state_dict(torch.load(model_path, map_location=device))
        preds = predict(model, device, test_loader)
        preds_sum += preds
    preds /= n_splits
    preds *= 100
    
    #test_df = utils.ensemble(test_df, 2, 'MGMT_value')
    test_df['Pawpularity']=preds

    test_df.to_csv(os.path.join(output_dir, 'submission.csv'))  
    test_df.to_csv('submission.csv')

    
    logger.info('::: success :::\n\n\n')

if __name__ == "__main__":
    run()
