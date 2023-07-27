""" main.py

Primary driver to define and execute experiments to train Evidential Distributions in Continuous Time. 
This example script sets up a demonstration of EDICT development for synthetically created irregular time series.

To run this script execute the following command:

    python main.py -config edict_base_syn

"""
import numpy as np
from tqdm import tqdm
import click
import os
import yaml

import torch

from data_utils import define_dataset
from model_utils import define_model
from losses import evidential_regression_loss

import hashlib
from utils import plot_trained_model, run_loop
from calibration import calc_calibration

@click.command()
@click.option('--config', '-c', default="edict_base_syn")
@click.option('--options', '-o', multiple=True, nargs=2, type=click.Tuple([str, str]))
@click.option('--demo', '-d', is_flag=True, help="Runs through pre-trained model and constructs figures demonstrating model performance on generated data")
def main(config, log_online, options, demo):
    """Using the specified configuration, train a sequentially based evidential regression model"""

    dir_path = os.path.dirname(os.path.realpath(__file__))
    params = yaml.safe_load(open(os.path.join(dir_path, f'configs/{config}.yaml')))

    # Replace configuration parameters by command line provided 'options'
    for opt in options:
        assert opt[0] in params
        dtype = type(params[opt[0]])
        if dtype == bool:
            new_opt = False if opt[1] != 'True' else True
        else:
            new_opt = dtype(opt[1])
        params[opt[0]] = new_opt

    params['exp_name'] = f"{params['exp_name']}{params['seed']}_{hashlib.md5(str(params).encode('utf-8')).hexdigest()[:6]}_{os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else ''}"
    save_dir = params.get('save_dir', './runs/')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    ckpt_fname = os.path.join(save_dir, params['exp_name'], "checkpoint.pt")
    local_save_dir = os.path.join("runs", params['exp_name'])
    if not os.path.isdir(local_save_dir):
        os.makedirs(local_save_dir, exist_ok=True)

    # Initialize the model for training the model primitives
    model = define_model(params, device)
    clip_value = params['grad_clip_value']  # Extract the clipping value clip_value = 1.5
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    if demo:
        print(f"Demo Mode -- Loading pretrained model {params['exp_name']} from {'runs/'+params['exp_name']} based on configuration file")
        plot_trained_model(model, params, model_dir=ckpt_fname, save_dir=local_save_dir, format='png')
        exit()

    # Extract the experiment name, also used to define the directory where the artifacts will be saved...
    exp_name = params['exp_name']

    print('Parameters')
    for key in params:
        print(key, params[key])
    print('=' * 30)
    print(f"Beginning experiment: {exp_name}")

    # Set the seeds
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    rng = np.random.RandomState(params['seed'])

    # Initialize the dataloaders for training and validation
    train_loader, val_loader, __, params = define_dataset(params, add_noise=None, device=device)

    # Dump the set of parameters to the artifact folder--> dumping after dataset creation in case there was a new dataset constructed and there
    # are a different number of sequences recorded than initially set in the original configuration file.
    with open(f"{save_dir}/{exp_name}/config.yaml", "w") as f:
        __ = yaml.dump(params, f, sort_keys=False, default_flow_style=False)

    with open(f"runs/{exp_name}/config.yaml", "w") as f:
        __ = yaml.dump(params, f, sort_keys=False, default_flow_style=False)

    # Define loss functions
    if params['output_type'] == 'evidential_multivariate':
        loss_fn = evidential_regression_loss(params)
    else:
        loss_fn = torch.nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lrn_rt'])

    # Best val_loss
    best_val_loss = 1e8

    max_num_epochs = params['num_epochs']

    # Run the training/validation loop
    for i_epch in range(max_num_epochs):
        model.train()
        for i, batch in tqdm(enumerate(train_loader)):
            run_loop(model, batch, params['model_type'], device, loss_fn, optimizer, i_epch, mode='train', method=params['dist_type'], T=params['max_time'])

        if i_epch % params['val_freq'] == 0:
            with torch.no_grad():
                model.eval()
                for _, batch_val in enumerate(val_loader):
                    val_loss = run_loop(model, batch_val, params['model_type'], device, loss_fn, optimizer, i_epch, mode='validation', method=params['dist_type'], T=params['max_time'])

                    print(f"Mean validation loss at epoch {i_epch}: {val_loss:.5f}")

                    df, calib_results = calc_calibration(params, model, 'val', device, prefix = 'val/')

                    # If best_val_loss, save the model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            'epoch': i_epch,
                            'model': model.state_dict(),
                            'val_loss': best_val_loss,
                            'calib_df': df
                        }, ckpt_fname)
                        print(f"Saved model into {ckpt_fname}.")

    print(f"Last validation loss: {val_loss:.5f}")

    if 'dataset_name' == 'syn_data':
        print(f"Evaluating trained model from experiment {params['exp_name']} on newly generated test conditions.")
        plot_trained_model(model, params, model_dir=ckpt_fname, save_dir=local_save_dir, format='png')

    model.load_state_dict(torch.load(ckpt_fname)['model'])
    model.eval()
    model = model.to(device)

    test_df, test_calib_results = calc_calibration(params, model, 'test', device, prefix = 'best_')
    test_df.to_csv(os.path.join("runs", params['exp_name'], 'test_calib.csv'))
        
if __name__ == "__main__":
    main()