"""
Utility and Helper functions for use to facilitate modeling the temporal evolution of an irregular timeseries

Notes:
 - 
"""

############################################
#           IMPORTS and DEFINITIONS
############################################
import os, math
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader

from data_utils import ODE_Dataset, custom_collate_fn, extract_from_path, syn_data_sample
from losses import niw_nll, log_normal_nll
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, brier_score_loss

### PLOTTING FOR SYNTHETIC DATA (Early stages)
def plot_NIW_data(x_sequence, train_cutoff, mu, psi, lmbda, nu):

    # Compute the "variance" of the different predictions
    # Essentially pulling off the sqrt of the diagonal of the covariance matrix
    times = x_sequence[:, 0]
    feats = x_sequence[:, 1:]

    dim = feats.shape[-1]

    prediction = mu.squeeze().numpy()
    # Compute the expected variance of the NIW, pull off the diagonal of the covarance matrix
    niw_var = torch.diagonal(psi / (lmbda * (nu - dim - 1)).unsqueeze(-1), dim1=-2, dim2=-1)
    niw_var = np.sqrt(np.clip(niw_var.detach().numpy(), -100, 100)).squeeze()  # Compute the square root of the component variances...

    colors = sns.color_palette(n_colors=feats.shape[-1])

    figs = []

    for ii in range(dim):
        fig = plt.figure()
        # Plot the observed data (as a curve)
        if train_cutoff is None:
            plt.vlines(times[-1], -2.5, 2.5, colors='black', linestyles='dotted')
        else:
            plt.vlines(times[train_cutoff], -2.5, 2.5, colors='black', linestyles='dotted')
        plt.scatter(times, feats[:, ii], lw=2, color=colors[ii])
        plt.plot(times, prediction[:, ii], lw=2, color=colors[ii])
        for k in np.linspace(0, 4, 4):
            plt.fill_between(times, (prediction[:,ii] - k*niw_var[:, ii]), (prediction[:,ii] + k*niw_var[:, ii]), alpha=0.3, edgecolor=None, facecolor=colors[ii], linewidth=0, zorder=1)

        plt.ylim([-2.5, 2.5])
        plt.xlabel('Time')

        figs.append(fig)

    # Track the variance over the missing portion of the validation sequence...
    var_figs = []
    for ii in range(dim):
        var_fig = plt.figure()
        plt.scatter(times, niw_var[:, ii], lw=2, color='gray', alpha=0.4)
        # Gather just the variance for this dimension corresponding to only the missing interval
        # Extract the elements of the features that are missing...
        missing_mask = np.isnan(feats[:, ii])

        if np.sum(missing_mask) > 0:  # Account for fully observed feature
            msk_times = times[missing_mask]
            msk_var = niw_var[missing_mask, ii]
            plt.scatter(msk_times, msk_var, lw=2, color=colors[ii])

        plt.ylim([-0.1, 2])
        plt.xlim([-0.5, np.max(times)])
        plt.ylabel('Variance')
        plt.xlabel('Time')

        var_figs.append(var_fig)

    return figs, var_figs

def plot_MSE_reconstruction(x_sequence, train_cutoff, prediction):

    # Compute the "variance" of the different predictions
    # Essentially pulling off the sqrt of the diagonal of the covariance matrix
    times = x_sequence[:, 0]
    feats = x_sequence[:, 1:]

    if train_cutoff is None:
        train_cutoff = times[-1]

    dim = feats.shape[-1]

    colors = sns.color_palette(n_colors=feats.shape[-1])

    figs = []

    for ii in range(dim):
        fig = plt.figure()
        # Plot the observed data (as a curve)
        if train_cutoff is None:
            plt.vlines(times[-1], -2.5, 2.5, colors='black', linestyles='dotted')
        else:
            plt.vlines(times[train_cutoff], -2.5, 2.5, colors='black', linestyles='dotted')
        plt.plot(times, feats[:, ii], '--', lw=2, color=colors[ii])
        plt.plot(times, prediction[:, ii], lw=2, color=colors[ii])
        
        plt.ylim([-2.5, 2.5])
        plt.xlabel('Time')

        figs.append(fig)

    return figs

def plot_NIW_online_sampling(data, mask, full_obs, obs_idx, t_vec, mean, var, times, time_ptr, query_time, max_time, plt_idx, save_dir):
    """Plot incrementally sampled data..."""

    plt.rcParams.update({'font.size': 22})

    # Expand the times vector to align with the length of obs_idx
    exp_times = np.zeros(len(obs_idx))
    for i, time in enumerate(times):
        exp_times[time_ptr[i]:time_ptr[i+1]] = time

    # extract the index of fully observed batch
    full_obs_idx = full_obs.index.unique().values

    if not isinstance(plt_idx, list):
        plt_idx = [plt_idx]

    for idx in plt_idx:

        q_idx = full_obs_idx[idx]
    
        # Isolate the fully_observed features
        df_i = full_obs.loc[q_idx]
        df_i = df_i[df_i.Time <= query_time].copy()  # Cut down to only the times we have observed

        # Extract the datapoints that correspond to the sequence we're plotting (plt_idx)
        obs_msk = obs_idx == idx
        msk_M = mask[obs_msk].type(torch.bool).numpy()
        msk_X = data[obs_msk].numpy()
        msk_times = exp_times[obs_msk]

        # Generate figure!

        mu = mean[:, idx, :].squeeze()
        v = var[:, idx, :].squeeze()
    
        up_2 = mu + 1.96 * torch.sqrt(v)
        down_2 = mu - 1.96 * torch.sqrt(v)

        fill_colors = sns.color_palette(n_colors=msk_X.shape[-1])
        line_colors = sns.color_palette(n_colors=msk_X.shape[-1])
        colors = sns.color_palette(n_colors=msk_X.shape[-1])

        plt_fname = os.path.join(save_dir, f"featAcq_t{query_time}_sample{idx}.png")
        plt.figure()
        for dim in range(msk_X.shape[-1]):
            plt.fill_between(x=t_vec, 
                            y1=down_2[:, dim].numpy()- 2*dim + 1,
                            y2=up_2[:,dim].numpy()- 2*dim + 1,
                            facecolor=fill_colors[dim],
                            edgecolor=None, linewidth=0,
                            alpha=0.35, zorder=1)
            plt.plot(t_vec, mu[:,dim].numpy()- 2*dim + 1, color=line_colors[dim], linewidth=2, zorder=2, label=f"Dimension {dim+1}")
            plt.scatter(msk_times[msk_M[:,dim]], msk_X[msk_M[:,dim],dim]- 2*dim + 1, color=colors[dim], edgecolors= 'k', alpha=0.75, s=60, zorder=3)
            plt.plot(df_i.Time, df_i[f"Value_{dim+1}"]- 2*dim + 1, ":", color=colors[dim], linewidth=1.5, alpha=0.8, label="_nolegend_", zorder=3)

        plt.xlim([0, max_time+2.5])
        plt.xlabel("Time")
        plt.ylabel("Prediction")
        plt.grid()
        plt.title(f"Query at t={query_time}")
        plt.savefig(plt_fname, dpi=500)
        plt.close()

def plot_trained_model(model, params, model_dir=None, save_dir=None, format='pdf', device='cpu'):
    """
    Using the predetermined parameters file for the run experiment:
      - load the trained model
      - generate some held out test data (or load some in the case of MIMIC)
      - plot model fit + uncertainty region around the data observations
    """
    style = 'fill'
    dataset_name = params.get('dataset_name', 'syn_data')
    dataset_dir = params.get('dataset_dir', f"dataset.csv")
    T = params.get('max_time', 4*math.pi)
    delta_t = params.get('delta_t', 0.05)
    sample_rate = params.get('sample_rate', 2)
    dual_sample_rate = params.get('dual_sample_rate', 0.2) + 0.25  # Making eval slightly sparser...

    if save_dir is None:
        save_dir = '/'.join(model_dir.split('/')[:-1])

    # Load the pretrained weights to the model
    model.load_state_dict(torch.load(model_dir)['model'])
    model.to(device)
    model.eval()

    plt.rcParams.update({'font.size': 22})

    # Generate/Extract some test data
    N = 10
    T *= 1.5  # Expand the time horizon beyond what's been evaluated previously
    if dataset_name == 'syn_data':
        obs_df, full_data = syn_data_sample(T=T, dt=delta_t, N=N, sample_rate=sample_rate, dual_sample_rate=dual_sample_rate)
    else:
        raise ValueError("Dataset name is not recognized")

    # Then construct a dataset for evaluating
    data = ODE_Dataset(panda_df=obs_df)
    dl = DataLoader(dataset=data, collate_fn=custom_collate_fn, shuffle=False, batch_size=1)

    # Loop through each sequence, run through the model and produce a figure output (would be great to tie this to plot_NIW_data()...)
    with torch.no_grad():
        for sample, batch in enumerate(dl):
            if sample > 10:
                break
            times = batch['times']
            time_ptr = batch['time_ptr']
            X = batch['X']
            M = batch['M']
            obs_idx = batch['obs_idx']
            pat_idx = batch['pat_idx']
            cov = batch['cov']

            observations = X.detach().numpy()

            if params['dist_type'] == 'niw':
                _, _, t_vec, mu_vec, lmb_vec, psi_vec, nu_vec, _, _, _ = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=T, cov=cov, return_path=True, pat_idx=pat_idx)
                v = (torch.exp(psi_vec) / (lmb_vec * (nu_vec - mu_vec.shape[-1] -1)))  # The expected variance of the NIW distribution
            else:
                _, _, t_vec, p_vec, _, _, _ = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=T, cov=cov, return_path=True, pat_idx=pat_idx)

                mu_vec, logvar = torch.chunk(p_vec, 2, dim=-1)
                v = torch.exp(0.5*logvar)

            up = mu_vec + 1.96 * torch.sqrt(v)
            down = mu_vec - 1.96 * torch.sqrt(v)

            plots_dict = dict()
            plots_dict['t_vec'] = t_vec
            plots_dict['up'] = up.numpy()
            plots_dict['down'] = down.numpy()
            plots_dict['mu'] = mu_vec.numpy()
            plots_dict['observations'] = observations
            plots_dict['mask'] = M.cpu().numpy()

            # Reduce unnecessary dims...
            up = up.squeeze()
            down = down.squeeze()
            mu = mu_vec.squeeze()

            fill_colors = sns.color_palette(n_colors=mu_vec.shape[-1])
            line_colors = sns.color_palette(n_colors=mu_vec.shape[-1])
            colors = sns.color_palette(n_colors=mu_vec.shape[-1])

            ## Trajectory ID
            if isinstance(full_data, pd.DataFrame):
                df_i = full_data.query(f"ID == {sample}")

            # eval_type = eval_type_names[sample % num_eval_types]

            plt.figure()
            if style == 'fill':
                for dim in range(mu_vec.shape[-1]):
                    plt.fill_between(x=t_vec, 
                                    y1=down[:, dim].numpy() - 2*dim + 1,
                                    y2=up[:,dim].numpy() - 2*dim + 1,
                                    facecolor=fill_colors[dim],
                                    edgecolor=None, linewidth=0,
                                    alpha=0.35, zorder=1)
                    plt.plot(t_vec, mu[:,dim].numpy() - 2*dim + 1, color=line_colors[dim], linewidth=2, zorder=2, label=f"Dimension {dim+1}")
                    observed_idx = np.where(plots_dict["mask"][:, dim]==1)[0]
                    plt.scatter(times[observed_idx], observations[observed_idx,dim] - 2*dim + 1, color=colors[dim], edgecolors= 'k', alpha=0.75, s=60, zorder=3)
                    plt.plot(np.linspace(0, T, int(T//delta_t)), full_data[sample,:,dim]- 2*dim + 1, '--', color=colors[dim], linewidth=1.5, alpha=0.8, label="_nolegend_", zorder=3)
            else:
                for dim in range(mu_vec.shape[-1]):
                    plt.plot(t_vec, up[:,dim].numpy(),"--", color="red", linewidth=2)
                    plt.plot(t_vec, down[:,dim].numpy(),"--", color="red",linewidth=2)
                    plt.plot(t_vec, mu_vec[:,dim].numpy(), color=colors[dim], linewidth=2)
                    observed_idx = np.where(plots_dict["mask"][:, dim]==1)[0]
                    plt.scatter(times[observed_idx], observations[observed_idx,dim], color=colors[dim], alpha=0.5, s=60)
                    plt.plot(df_i.Time, df_i[f"Value_{dim+1}"], ":", color=colors[dim], linewidth=1.5, alpha=0.8)

            plt.xlabel("Time")
            plt.grid()
            # plt.legend(loc="lower right")
            # plt.ylim([-2.5, 2.5])
            plt.ylabel('Prediction')
            # Construct filename based on exp_name directory (can pull from checkpoint fname above...)
            # Adjust filename based on 'sample' (we can construct a mapping!)
            # fname = '/'.join(checkpoint_fname.split('/')[:-1])+f"/eval_{eval_type}_sample{sample}_{style}.{format}"
            fname = save_dir+f"/eval_sample{sample}_{style}.{format}"
            plt.tight_layout()
            plt.savefig(fname, dpi=500)
            plt.close()
            print(f"Saved sample into '{fname}'.")


#############################################
#   MODEL TRAINING AND EVALUATION F'ns
#############################################

def run_gruode_loop(model, batch, device, loss_fn, optimizer, epoch, delta_t=0.05, T=12, method='niw', mode='train'):

    # Extract information from the batch
    times = batch['times']
    time_ptr = batch['time_ptr']
    X = batch['X'].to(device)
    M = batch['M'].to(device)
    obs_idx = batch['obs_idx']
    pat_idx = batch['pat_idx']  # Extracting the core patient index for debugging purposes to see if it's the same trajectory running to NaN each time...
    cov = batch['cov'].to(device)

    y = batch['y']

    if mode == 'train':
        optimizer.zero_grad()
        _, loss, _, _, _ = model(times, time_ptr, X, M, obs_idx, delta_t = delta_t, T=T, cov=cov, pat_idx=pat_idx)
        loss.backward()
        optimizer.step()
        
        
    else:
        X_val = batch['X_val'].to(device)
        M_val = batch['M_val'].to(device)
        times_val = batch['times_val']
        times_idx = batch['index_val']

        if method == 'niw':
            _, loss, t_vec, mu_vec, lmb_vec, psi_vec, nu_vec, _, _, _ = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=T, cov=cov, pat_idx=pat_idx, return_path=True)
        else:
            _, loss, t_vec, p_vec, _, _, _ = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=T, cov=cov, pat_idx=pat_idx, return_path=True)

        # loss, temp_reg, error_reg = loss
        t_vec = np.around(t_vec, str(delta_t)[::-1].find('.')).astype(np.float32)  # Round floating points error in the time vector

        if method == 'niw':
            m = extract_from_path(t_vec, mu_vec, times_val, times_idx)
            lmb_val = extract_from_path(t_vec, lmb_vec, times_val, times_idx)
            psi_val = extract_from_path(t_vec, psi_vec, times_val, times_idx)
            nu_val = extract_from_path(t_vec, nu_vec, times_val, times_idx)

            val_nll = (niw_nll(X_val, m, psi_val, lmb_val, nu_val, M_val)).sum().cpu() / M_val.sum().cpu().numpy()
        else:
            p_val = extract_from_path(t_vec, p_vec, times_val, times_idx)
            m, v = torch.chunk(p_val, 2, dim=1)

            val_nll = (log_normal_nll(X_val, m, v, M_val)).sum().cpu() / M_val.sum().cpu().numpy()

        val_mse = (torch.pow(X_val-m, 2)*M_val).sum().cpu() / M_val.sum().cpu().numpy()

        return loss.item()

def run_clf_loop(model, batch, device, loss_fn, optimizer, epoch, delta_t=0.05, T=12, method='niw', mode='train'):
    # Extract information from the batch
    times = batch['times']
    time_ptr = batch['time_ptr']
    X = batch['X'].to(device)
    M = batch['M'].to(device)
    obs_idx = batch['obs_idx']
    pat_idx = batch['pat_idx']  # Extracting the core patient index for debugging purposes to see if it's the same trajectory running to NaN each time...
    cov = batch['cov'].to(device)

    y = batch['y']

    logits = model(times, time_ptr, X, M, obs_idx, delta_t = delta_t, T=T, cov=cov, pat_idx=pat_idx)
    loss = model.get_loss(logits, y)
    if mode == 'train':
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    else:
        return loss.item()

def run_loop(model, batch, model_type, device, loss_fn, optimizer, epoch, mode='train', method='niw', **kwargs):

    if model_type == 'gruode':
        if mode == 'validation':
            val_loss = run_gruode_loop(model, batch, device, loss_fn, optimizer, epoch, mode=mode, method=method, **kwargs)
            return val_loss
        else:
            run_gruode_loop(model, batch, device, loss_fn, optimizer, epoch, mode=mode, method=method, **kwargs)
    elif model_type == "clf":
        if mode == 'validation':
            return run_clf_loop(model, batch, device, loss_fn, optimizer, epoch, mode=mode, method=method, **kwargs)
        else:
            run_clf_loop(model, batch, device, loss_fn, optimizer, epoch, mode=mode, method=method, **kwargs)
    else:
        raise ValueError("Provided model type is not implemented. Please use: 'ncde' or 'gruode'.")

def score_eval(y, pred_y):
    res = {
        'log_loss': log_loss(y, pred_y, eps = 1e-6)
    }

    if len(np.unique(y)) == 2:
        res['auroc'] = roc_auc_score(y, pred_y[:, 1]),
        res['brier'] = brier_score_loss(y, pred_y[:, 1])
        res['auprc'] = average_precision_score(y, pred_y[:, 1])
    
    return res