import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


def save_loss_figure(fold_i,
                     epochs,
                    losses_train,
                    losses_valid,
                    losses_rmse_train,
                    losses_rmse_valid,
                    save_dir
                    ):
    fig = plt.figure()
    plt.plot(epochs, losses_train, '-x', label = 'train')
    plt.plot(epochs, losses_valid, '-x', label='valid')


#    plt.plot(epochs, losses_rmse_train/100, '-o', label='train_rmse/100')
#    plt.plot(epochs, losses_rmse_valid/100, '-o', label='valid_rmse/100')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend()
    fig.savefig(f'{save_dir}/loss_fold{fold_i}.png')

def save_result_csv(
    fold_i, debug, model_name, loss_name, best_loss, best_loss_rmse,  comment, save_dir
):
    df = pd.DataFrame({
        'run_name': [save_dir.split('hydra_outputs/')[1]],
        'debug' : [debug],
        'fold' : [fold_i ],
        'model_name' : [model_name],
        'loss_name' : [loss_name],
        'best_loss' : [round(best_loss, 6)],
        'best_loss_rmse' : [round(best_loss_rmse, 6)],
        'comment': [comment]
    })

    df.to_csv(f'{save_dir}/result_fold{fold_i}.csv', index=False)