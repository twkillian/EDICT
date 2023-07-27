import pdb
import os, sys, math
import yaml

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import scipy
from data_utils import ODE_Dataset, custom_collate_fn


def calc_calibration(params, model, sset, device, prefix = '', num_seq = None):
    """Using the specified configuration, evaluate a sequentially based evidential regression model and save off the corresponding data points"""
    del sset
    model.eval()
    
    # Initialize the evaluation dataset
    N = params.get('num_sequences', 10000)
    dataset_name = params.get('dataset_name', 'syn_data')
    dataset_dir = params.get('dataset_dir', f"dataset.csv")
    max_time = params.get('max_time', 4*math.pi)
    T_val = params.get('max_time_val', 3*math.pi)
    delta_t = params.get('delta_t', 0.05)
    batch_size = params.get('tr_batch_size', 500)

    _, test_idxs = train_test_split(np.arange(N), test_size=0.2)

    if num_seq is None:
        test_idxs = test_idxs
    else:
        test_idxs = np.random.choice(test_idxs, replace=False, size=num_seq)  # Samples the indices of the sequences we want to pull
    
    if dataset_name  == 'syn_data':
        cov_file = None
        label_file = None
    else:
        raise ValueError("Dataset name is not recognized")

    data = ODE_Dataset(csv_file=dataset_dir, cov_file=cov_file, label_file=label_file, idx=test_idxs, T_val=T_val, calibration_test=True)

    dl = DataLoader(dataset=data, collate_fn=custom_collate_fn, shuffle=False, batch_size=batch_size)

    # Loop through each sequence, run through the model and produce a figure output (would be great to tie this to plot_NIW_data()...)
    with torch.no_grad():
        for sample, batch in enumerate(dl):
            if sample > 10:
                break
            times = batch['times']
            time_ptr = batch['time_ptr']
            # Values used to create the distribution
            X = batch['X'].to(device)
            M = batch['M'].to(device)
            obs_idx = batch['obs_idx']
            pat_idx = batch['pat_idx']
            cov = batch['cov'].to(device)

            # Values used to evaluate the interpolation
            X_interp = batch['X_interp']
            M_interp = batch['M_interp']
            times_interp = batch['times_interp']
            index_interp = batch['index_interp']

            # Values used to evaluate the extrapolation
            X_extrap = batch['X_val'].to(device)
            M_extrap = batch['M_val'].to(device)
            times_extrap = batch['times_val']
            index_extrap = batch['index_val']

            if params['dist_type'] == 'niw':
                _, _, t_vec, mu_vec, lmb_vec, psi_vec, nu_vec, h_vec, _, _ = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=max_time, cov=cov, return_path=True, pat_idx=pat_idx)
                v = (torch.exp(psi_vec) / (lmb_vec * (nu_vec - mu_vec.shape[-1] -1)))  # The expected variance of the NIW distribution
            else:
                _, loss, t_vec, p_vec, _, _, _ = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=max_time, cov=cov, return_path=True, pat_idx=pat_idx)

                mu_vec, logvar = torch.chunk(p_vec, 2, dim=-1)
                v = torch.exp(0.5*logvar)

            # Reduce unnecessary dims...
            mu = mu_vec.squeeze()
            v = v.squeeze()

            num_dims = mu_vec.shape[-1]
            num_seq = len(np.unique(obs_idx))

            # Loop over each segment and place into a dataframe that can then be written to CSV...
            cols = ['ID', 'Time', 'Dim', 'true_value', 'pred_mean','pred_var','method', 'Extrapolation']
            float_cols = ['ID','Time', 'Dim', 'true_value', 'pred_mean', 'pred_var', 'Extrapolation']
            method = 'NIW' if params['dist_type'] == 'niw' else 'GOB'
            df = pd.DataFrame(columns=cols)
            for i_seq in range(num_seq):
                
                seq_mean = mu[:,i_seq,:].detach().cpu().numpy()
                seq_var = v[:, i_seq, :].detach().cpu().numpy()
                
                for i_extp in range(2): # switch between extrapolation and interpolation
                # # Sample times from the non-observed time points (~10% of the possible entries... Can do a set difference between obs_df.Time and full_df.Time?)
                # # These will be the timepoints that we evaluate across. We can potentially resample lots of times to get our CIs?
                # sample_idx = np.random.choice(N_t, int(0.1*N_t), replace=False)
                # sample_times = delta_t * sample_idx
                     
                    if i_extp == 0: 
                        seq_idx = index_interp == i_seq
                        gnd_truth = X_interp[seq_idx,  :].detach().cpu().numpy()
                        gnd_masks = M_interp[seq_idx, :].detach().cpu().numpy()
                        gnd_times = times_interp[seq_idx]
                    else:
                        seq_idx = index_extrap == i_seq
                        gnd_truth = X_extrap[seq_idx,  :].detach().cpu().numpy()
                        gnd_masks = M_extrap[seq_idx, :].detach().cpu().numpy()
                        gnd_times = times_extrap[seq_idx]
                

                    # The GOB backbone doubles up the times where there's been an observation so we need to index through `t_vec` 
                    # based on `sample_times` to gather the predicted mean and variance at these points... 
                    # Additionally, we'll get a sense of whether we're interpolating/extrapolating based on these times as well
                    for ii, i_time in enumerate(gnd_times):
                        # Get the query_index in these expanced vectors output from the GOB backbone
                        # Since the data is held out, there's no guarantee (within the tolerances of np.isclose)
                        #  that this code will provide an index. Converting to integers by dividing by dt works.
                        #  Relaxing with an index on either side of the i_time//delta_t construction... Was still running into issues with the activity dataset...
                        q_idx = np.where(np.isclose(t_vec//delta_t, i_time//delta_t, rtol=1, atol=1))[0][0]
                        # Get the time specific mask for this observation, will be used to isolate only the observed features
                        obs_mask = gnd_masks[ii, :].astype(bool)
                        # Get the ground truth vector for this time
                        truth = gnd_truth[ii, :][..., None][obs_mask]
                        # Get the predicted mean
                        pred_mean = seq_mean[q_idx, :][..., None][obs_mask]
                        # Get the predicted variance
                        pred_var = seq_var[q_idx, :][..., None][obs_mask]

                        # Aggregate the data for this time step and add it to the larger dataframe
                        individual_data = pd.DataFrame(np.concatenate((i_seq*np.ones((num_dims, 1))[obs_mask], i_time*np.ones((num_dims, 1))[obs_mask], np.arange(num_dims)[...,None][obs_mask], truth, pred_mean, pred_var, np.array([method]*num_dims)[...,None][obs_mask], i_extp*np.ones((num_dims, 1))[obs_mask]), 1), columns=cols)
                        df = pd.concat([df, individual_data], axis=0, join='outer', ignore_index=True)

        df[float_cols] = df[float_cols].astype(np.float32)

        df.reset_index(drop=True, inplace=True)
      
        df['z'] = (df['true_value'] - df['pred_mean'])/df['pred_var']
        df['cdf_val'] = scipy.special.ndtr(df['z'].values)
        df['abnormal_val'] =  np.abs(0.5 - scipy.special.ndtr(df['z'].values))
        res = {}
        
        for extrap in [0, 1]:
            df_sub = df[df['Extrapolation'] == extrap]
            x_grid = np.linspace(0, 1, 20)
            coverage_vals =  []
            for x in x_grid:  
                coverage_per_id = df_sub.groupby('ID').apply(lambda y: (y['abnormal_val'] <= x/2).sum()/len(y))
                coverage_vals.append(coverage_per_id.mean())
            res = {**res, **{f'{prefix}ece_{"extrap" if extrap == 1 else "interp"}': np.abs(np.array(coverage_vals) - x_grid).sum()/len(x_grid)}}
            res = {**res, **{f'{prefix}mse_{"extrap" if extrap == 1 else "interp"}': ((df_sub['true_value'] - df_sub['pred_mean'])**2).mean()}}
        
        return df, res
    