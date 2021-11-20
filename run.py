import warnings

warnings.filterwarnings("ignore")

import os
import gc
import hydra
import torch
import logging
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_USER
import sklearn.model_selection as sms

from src import utils
from src import configuration as C
from src import models
from src.early_stopping import EarlyStopping
from src.train import train_simple
from src.eval import valid
import yaml
from fastprogress import master_bar
from src.mlflow_writer import MlflowWriter

logger = logging.getLogger(__name__)
config_path = "./config"
config_name = "run_config.yaml"


@hydra.main(config_path=config_path, config_name=config_name)
def run(cfg: DictConfig) -> None:

    # --------------- init logger,etc... -----------------
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")
    writer = MlflowWriter(cfg["globals"]["ex_name"])
    tags = {
        MLFLOW_RUN_NAME: f"{cfg['dataset']['name']} - {cfg['model']['name']}",
        MLFLOW_USER: cfg["globals"]["user"],
    }
    writer.create_new_run(tags)

    global_params = cfg["globals"]
    utils.set_seed(global_params["seed"])
    device = C.get_device()

    # -------------- init data ------------------
    df, datadir = C.get_metadata(cfg)
    if cfg["globals"]["debug"]:
        logger.info("::: set debug mode :::")
        cfg = utils.get_debug_config(cfg)
        df = utils.get_debug_df(df)

    # ToDo: 層化抽出
    train_df, valid_df = sms.train_test_split(df, test_size=0.2, shuffle=True)
    train_df = utils.min_max_normalize(train_df, "Pawpularity", 0, 100)
    valid_df = utils.min_max_normalize(valid_df, "Pawpularity", 0, 100)

    logger.info(f"train_df: {train_df.shape}")
    logger.info(f"valid_df: {valid_df.shape}")

    train_loader = C.get_loader(train_df, datadir, cfg, "train")
    valid_loader = C.get_loader(valid_df, datadir, cfg, "valid")

    # ------------ init model --------------------
    model = models.get_model(cfg).to(device)

    # ------------ init optimizer --------------
    optimizer = C.get_optimizer(model, cfg)
    scheduler = C.get_scheduler(optimizer, cfg)

    # ----------- init loss, ES---------------
    loss_func = C.get_criterion(cfg)
    early_stopping = EarlyStopping(**cfg["callback"]["early_stopping"], verbose=True)

    # ---------- init tracker ------------
    best_loss = 0
    losses_train = []
    losses_valid = []

    epochs_test = []
    losses_test = []

    n_epoch = cfg["globals"]["num_epochs"]
    x_bounds = [0, n_epoch]
    y_bounds = [0, 10]

    # ------- for epoch -----------
    mb = master_bar(range(n_epoch))
    for epoch in mb:
        logger.info(f"EPOCH:{epoch}")
        loss_train = 0
        loss_valid = 0

        # --------- train loop -----------
        loss_train = train_simple(
            model, device, train_loader, optimizer, scheduler, loss_func, mb
        )
        writer.log_metric_step("train loss", loss_train, step=epoch)
        losses_train.append(loss_train)

        # --------- valid loop ----------
        loss_valid, y_pred, y_true = valid(model, device, valid_loader, loss_func)
        writer.log_metric_step("valid loss", loss_valid, step=epoch)
        losses_valid.append(loss_valid)

        # --------- inform result --------
        logger.info(f"loss_train: {loss_train:.4f}, loss_valid: {loss_valid:.4f}")
        logger.info(f"y_pred : { [round(y, 1) for y in y_pred[:10]] }")
        logger.info(f"y_true : { [round(y ,1) for y in y_true[:10]] }")

        # -------- write loss curve ------
        t = np.arange(epoch)
        graphs = [[t, losses_train], [t, losses_valid]]
        mb.update_graph(graphs, x_bounds, y_bounds)
        mb.write( f"EPOCH: {epoch:02d}, Training loss: {loss_train:10.5f}, Validation loss: {loss_valid:10.5f}" )

        # --------- loss check ------------
        is_update = early_stopping(loss_valid,model,debug=False)
        if is_update:
            best_loss = loss_valid
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    logger.info(f"best valid loss: {best_loss:.6f}")

    # --------- output logs -------------
    losses_df = pd.DataFrame()
    losses_df["loss_train"] = losses_train
    losses_df["loss_valid"] = losses_valid
    losses_df.to_csv("losses.csv")
    writer.log_artifact("losses.csv")

    losses_test_df = pd.DataFrame()
    losses_test_df["epochs"] = epochs_test
    losses_test_df["loss_test"] = losses_test
    losses_test_df.to_csv("losses_test.csv")
    writer.log_artifact("losses_test.csv")

    writer.log_artifact("weight.pth")

    # ----------clear cache ---------
    del train_loader
    del valid_loader
    del model
    del optimizer
    del scheduler
    gc.collect()
    torch.cuda.empty_cache()

    # ------------- end ----------
    logger.info("::: success :::\n\n\n")
    writer.set_terminated()


if __name__ == "__main__":
    yaml_path = os.path.join(config_path, config_name)

    with open(yaml_path, "r+") as f:
        cfg = yaml.safe_load(f)
        if cfg["globals"]["ex_name"] == None:
            ex_name = input("EXPERIMENT_NAME：")
            cfg["globals"]["ex_name"] = ex_name
        f.seek(0)
        yaml.safe_dump(cfg, f, sort_keys=False)

    run()

    cfg["globals"]["ex_name"] = None
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
