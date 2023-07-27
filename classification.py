import os
import yaml
import click

import numpy as np

import torch

import hashlib

from data_utils import define_dataset
from model_utils import define_model, clf_model, IPN, SeFTNetwork, GRUD
import utils

def run_clf_epoch(dl, model, clf, criterion, optimizer, delta_t=0.05, max_time=5.0, train=True, train_seq_model=False, device='cpu', model_type="gruode", dist_type='niw'):

    # Set up the sequential model for training, if needed
    if train_seq_model:
        model.train()
    
    epoch_loss, num_correct, total_num = 0, 0, 0
    clf_preds_all, ys = [], []

    for sample, batch in enumerate(dl):
        times = batch['times']        
        X = batch['X'].to(device)
        M = batch['M'].to(device)        
        pat_idx = batch['pat_idx']
        cov = batch['cov'].to(device)
        y = batch['y'].to(device).long()

        if 'time_ptr' in batch:
            time_ptr = batch['time_ptr']
            obs_idx = batch['obs_idx'].to(device)        

        # Encode the sequences using the trained sequential model
        if model_type == "gruode":
            with torch.no_grad():
                if dist_type == 'niw':
                    hT, _, _, _, _, _, _, _, _, _ = model(times, time_ptr, X, M, obs_idx, delta_t, T=max_time, cov=cov, pat_idx=pat_idx, return_path=True)
                else:
                    hT, _, _, _, _, _, _ = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=max_time, cov=cov, pat_idx=pat_idx, return_path=True)
        elif model_type in ['ipn', 'seft']:
            # If we're using other models, we need to pass grads thru classifier
            hT = model(times, time_ptr, X, M, obs_idx, delta_t, T=max_time, cov=cov, pat_idx=pat_idx)
        elif model_type == 'grud':
            hT = model(times, X, M, cov)
        else:
            print("wrong model type")

        if train:
            optimizer.zero_grad()
            clf_preds = clf(hT)
            clf_loss = criterion(clf_preds, y.squeeze())
            clf_loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                clf_preds = clf(hT)
                clf_loss = criterion(clf_preds, y.squeeze())
        
        epoch_loss += clf_loss.item()
        num_correct += torch.sum(torch.argmax(clf_preds.detach(), dim=-1) == y.squeeze()).cpu().item()
        total_num += clf_preds.shape[0]
        clf_preds_all.append(torch.softmax(clf_preds.detach().cpu(), dim = 1).numpy())
        ys.append(y.detach().cpu().numpy())

    return epoch_loss, sample+1, num_correct, total_num, np.concatenate(clf_preds_all), np.concatenate(ys)


@click.command()
@click.option('--exp_name', '-e', default='syn_data/niw_gruode_syn_1', type=str, help='specify the pretrained sequential model')
@click.option('--add_noise', '-n', default=-0.1, type=float, help='the maximum level of noise exponentially added to observations')
@click.option('--seed', '-s', default=0, type=int, help='Random seed for varying experimental initialization.')
@click.option('--device', default='cuda', type=str, help='Force execution on a specific device...')
@click.option('--use_seq', is_flag=True, help="If we're using a sequential model that has been pretrained. Changes where configuration files comes from")
@click.option('--eval', is_flag=True, help="If true, classification model has already been trained and we'll only be operating on the test data")
@click.option('--reweight_obs', is_flag=True, help="If true, we will reweight the observations according to the predicted distribution when evaluating on the test set.")
@click.option('--use_pop_mean', is_flag=True, help="If true, we will reweight the observations according to the training population mean and std...")
@click.option('--options', '-o', multiple=True, nargs=2, type=click.Tuple([str, str]))
def run_classification(exp_name, add_noise, seed, device, use_seq, eval, reweight_obs, use_pop_mean, log_online, options):   
    # Initialize the artifacts for saving...
    training_losses, training_accuracies = [], []
    validation_losses, validation_accuracies = [], []
    testing_loss, testing_accuracy = [], []

    if add_noise < 0:
        add_noise = None

    # Load the configuration .yaml file from the provided experiment name
    if use_seq:
        exp_dir = os.path.join("runs", exp_name)
        params = yaml.safe_load(open(os.path.join(exp_dir, 'config.yaml')))
    else:
        exp_dir = os.path.join("configs", exp_name)
        params = yaml.safe_load(open(f'{exp_dir}.yaml'))

        # Update exp_name with the dataset directory prepended (for logging purposes)
        exp_name = f"{params['dataset_name'].split('_')[0]}/{exp_name}"
    

    # Replace configuration parameters by command line provided 'options'
    for opt in options:
        if opt[0] in params:
            dtype = type(params[opt[0]])
            if dtype == bool:
                new_opt = False if opt[1] != 'True' else True
            else:
                new_opt = dtype(opt[1])
            params[opt[0]] = new_opt
        else:  # Making assumption that the option provided is numeric (hack for now...)
            params[opt[0]] = int(opt[1])

    # Adding in the defaults so the hash key is the same for indexing purposes... 
    params['add_noise'] = None
    params['reweighting'] = False

    # Seq Model parameters
    dist_type = params.get('dist_type', 'niw')

    # Classifier parameters
    clf_lr = params.get('clf_lr', 0.001)
    clf_epochs = params.get('clf_num_epochs', 25)
    batch_size = params.get('tr_batch_size', 256)
    num_clf_layers = params.get('clf_num_layers', 2)
    clf_output_dims = params.get('clf_output_dims', 7)
    clf_val_freq = params.get('clf_val_freq',5)  # Run validation loop every K epochs
    hidden_state_dim = params.get('hidden_size', 50)
    clf_hidden_dim = params.get('clf_hidden', 50)


    # Establish directory for saving off artifacts
    if eval:
        job_add = f"_{hashlib.md5(str(params).encode('utf-8')).hexdigest()[:6]}_"
        addition = ''
        if ("edict_syn" in exp_name):
            search_key = f"clf_s{params['seed']}_hidden{clf_hidden_dim}_epochs{clf_epochs}_bs{batch_size}_lr{clf_lr}{addition}"
        else:
            search_key = f"clf_s{params['seed']}_hidden{clf_hidden_dim}_epochs{clf_epochs}_bs{batch_size}_lr{clf_lr}{addition}{job_add}"
        clf_dirs = next(os.walk(f"runs/{exp_name}/"))[1]
        eval_dir = [d for d in clf_dirs if search_key in d][0]
        clf_dir = os.path.join(f"runs/{exp_name}/", eval_dir)
    else:
        job_add = f"_{hashlib.md5(str(params).encode('utf-8')).hexdigest()[:6]}_{os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else ''}"
        addition = ''
        clf_dir = f"runs/{exp_name}/clf_s{params['seed']}_hidden{clf_hidden_dim}_epochs{clf_epochs}_bs{batch_size}_lr{clf_lr}{addition}{job_add}/"
    
    if not os.path.exists(clf_dir):
        os.makedirs(clf_dir)

    
    # We have to add these things after the directory identification because we'd changing the hash key...
    params['seed'] = seed
    params['add_noise'] = add_noise
    params['reweighting'] = reweight_obs
    params['use_pop_mean'] = use_pop_mean

    # Set the seeds
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    rng = np.random.RandomState(params['seed'])
    

    clf_ckpt_fname = os.path.join(clf_dir, "clf_checkpoint.pt")
    with open(os.path.join(clf_dir, "config.yaml"), "w") as f:
        __ = yaml.dump(params, f, sort_keys=False, default_flow_style=False)

    try:
        device = torch.device(device)
    except:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = params.get("dataset_name", "syn_data")
    delta_t = params.get('delta_t', 0.05)
    max_time = params.get('max_time')
    
    # Initialize training, validation and testing datasets
    if ('ipn' in exp_name):
        train_dl, val_dl, test_dl, params = define_dataset(params, add_noise, True, device) # Need to constrain the size of the test set output...
    else:
        train_dl, val_dl, test_dl, params = define_dataset(params, add_noise, device)

    try:
        pop_mean = train_dl.dataset.pop_mean
        pop_std = train_dl.dataset.pop_std
    except:
        pop_mean = None
        pop_std = None

    if use_pop_mean:
        params['pop_mean'] = torch.tensor(pop_mean, dtype=torch.float32).to(device)
        params['pop_std'] = torch.tensor(pop_std, dtype=torch.float32).to(device)
    else:
        params['pop_mean'] = None
        params['pop_std'] = None

    # Extract the input dimension for defining models
    if 'input_size' in params:
        input_size = params.get('input_size')
    else:
        input_size = val_dl.dataset.variable_num

    # Define a classification model that takes the hidden state of the sequential model and predicts the class
    clf = clf_model(hidden_state_dim, clf_hidden_dim, clf_output_dims)
    clf = clf.to(device)

    # Define the sequential model and load the pre-trained parameters
    if use_seq:
        seq_model = define_model(params, device)
        seq_model.to(device)
        seq_model.load_state_dict(torch.load(os.path.join(exp_dir, "checkpoint.pt"))['model'])
        seq_model.eval()

        # Define an optimizer over just the classifier
        if not eval:
            optimizer = torch.optim.Adam(clf.parameters(), lr=clf_lr)
        else:
            optimizer = None

    elif params.get('model_type') == 'ipn':       
        nref = int(max_time//delta_t)
        seq_model = IPN(input_size, hidden_state_dim, nref).to(device)
        optimizer = torch.optim.Adam(list(clf.parameters()) + list(seq_model.parameters()), lr=clf_lr)
    elif params.get('model_type') == 'seft':
        seq_model = SeFTNetwork(params).to(device)
        optimizer = torch.optim.Adam(list(clf.parameters()) + list(seq_model.parameters()), lr=clf_lr)
    elif params.get('model_type') == 'grud':
        seq_model = GRUD(params, input_size).to(device)
        optimizer = torch.optim.Adam(list(clf.parameters()) + list(seq_model.parameters()), lr=clf_lr)
    
    criterion = torch.nn.CrossEntropyLoss()

    if eval:  # If we're evaluating, let's load a pretrained classifier
        if not use_seq:
            # Load the pre-trained "sequence model" [IPN, SeFT, GRU-D]
            seq_model.load_state_dict(torch.load(clf_ckpt_fname)['seq_model'])
            seq_model.eval()
        clf.load_state_dict(torch.load(clf_ckpt_fname)['model'])
        clf.eval()

    # Loop through training epochs with periodic validation
    if not eval:

        # Best val_loss
        best_val_loss = 1e8

        for epoch in range(clf_epochs):

            epoch_loss, num_batches, num_correct, total_num, _, _ = run_clf_epoch(train_dl, seq_model, clf, criterion, optimizer, 
                                                                            delta_t=params['delta_t'], max_time=params['max_time'], train=True, 
                                                                            train_seq_model= not use_seq, device=device, model_type=params['model_type'], dist_type=dist_type)


            training_accuracies.append(num_correct/total_num)
            training_losses.append(epoch_loss/num_batches)

            print(f"TRAINING: After epoch {epoch}, mean loss is: {epoch_loss/num_batches} with mean accuracy: {num_correct/total_num}")

            if epoch % clf_val_freq == 0:
                seq_model.eval()
                clf.eval()
                val_epoch_loss, val_num_batches, val_num_correct, val_total_num, _, _ = run_clf_epoch(val_dl, seq_model, clf, criterion, optimizer, 
                                                                                                delta_t=params['delta_t'], max_time=params['max_time'], train=False, 
                                                                                                train_seq_model=False, device=device, model_type=params['model_type'], dist_type=dist_type)


                validation_accuracies.append(val_num_correct/val_total_num)
                validation_losses.append(val_epoch_loss/val_num_batches)

                print(f"VALIDATION: After epoch {epoch}, mean loss is: {val_epoch_loss/val_num_batches} with mean accuracy: {val_num_correct/val_total_num}")
                # If best_val_loss, save the model
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    val_save_dict = {
                        'epoch': epoch,
                        'model': clf.state_dict(),
                        'val_loss': best_val_loss,
                        'train_acc': training_accuracies,
                        'train_loss': training_losses,
                        'val_acc': validation_accuracies,
                        }
                    if not use_seq:
                        val_save_dict['seq_model'] = seq_model.state_dict()
                    torch.save(val_save_dict, clf_ckpt_fname)
                    print(f"Saved model into {clf_ckpt_fname}.")
                clf.train()
                if not use_seq:
                    seq_model.train()

        clf.eval()
        seq_model.eval()
    
    state_dict = torch.load(clf_ckpt_fname)

    if 'seq_model' in state_dict:
        seq_model.load_state_dict(state_dict['seq_model'])

    clf.load_state_dict(state_dict['model'])

    clf.eval()
    seq_model.eval()

    print("EVALUATING THE CLASSIFIER ON THE TEST SET")
    test_epoch_loss, test_num_batches, test_num_correct, test_total_num, clf_preds, ys = run_clf_epoch(test_dl, seq_model, clf, criterion, optimizer, 
                                                                                        delta_t=params['delta_t'], max_time=params['max_time'], train=False, 
                                                                                        train_seq_model=False, device=device, model_type=params['model_type'], dist_type=dist_type)

    testing_accuracy.append(test_num_correct/test_total_num)
    testing_loss.append(test_epoch_loss/test_num_batches)
    
    print('#'*25)
    print(f"FINAL EVALUATION: Accuracy = {test_num_correct/test_total_num}")
    print('~#~'*10)

    # Construct a dictionary to save the classification artifacts
    save_dict = {**{'test_acc': testing_accuracy,
                'test_loss': testing_loss,
                'clf_preds': clf_preds,
                'y': ys},
                **utils.score_eval(ys, clf_preds)}
    
    eval_type = "_reweighting" if reweight_obs else ""
    use_pop = "_wPopMean" if use_pop_mean else ""
    clf_eval_fname = os.path.join(clf_dir, f"eval_n{add_noise}{eval_type}{use_pop}.pt")

    torch.save(save_dict, clf_eval_fname)
        
if __name__ == '__main__':
    run_classification()