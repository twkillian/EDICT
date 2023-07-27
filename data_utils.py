""" data_utils.py

This file contains all data preprocessing and generating functions, modules or classes

Notes:
 - 

"""
############################################
#                 IMPORTS
############################################
import os, sys, math
import numpy as np
import pandas as pd

import pdb

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader

from sklearn.model_selection import train_test_split

############################################
#           HELPER FUNCTIONS
############################################

def map_to_closest(input,reference):
    output = np.zeros_like(input)
    for idx, element in enumerate(input):
        closest_idx=(np.abs(reference-element)).argmin()
        output[idx]=reference[closest_idx]
    return(output)

def extract_from_path(t_vec, p_vec, eval_times, path_idx_eval):
    '''
    Takes :
    t_vec : numpy vector of absolute times length [T]. Should be ordered.
    p_vec : numpy array of means and logvars of a trajectory at times t_vec. [T x batch_size x (2xfeatures)]
    eval_times : numpy vector of absolute times at which we want to retrieve p_vec. [L]
    path_idx_eval : index of trajectory that we want to retrieve. Should be same length of eval_times. [L]
    Returns :
    Array of dimensions [L,(2xfeatures)] of means and logvar of the required eval times and trajectories
    '''
    #Remove the evaluation after the updates. Only takes the prediction before the Bayesian update. 
    t_vec, unique_index = np.unique(t_vec,return_index=True)
    p_vec = p_vec[unique_index,:,:]

    present_mask = np.isin(eval_times, t_vec)
    eval_times[~present_mask] = map_to_closest(eval_times[~present_mask],t_vec)

    mapping = dict(zip(t_vec,np.arange(t_vec.shape[0])))

    time_idx = np.vectorize(mapping.get)(eval_times)

    return(p_vec[time_idx,path_idx_eval,:])
    
def sort_array_on_other(x1,x2):
    """
    This function returns the permutation y needed to transform x2 in x1 s.t. x2[y]=x1
    """

    temp_dict = dict(zip(x1,np.arange(len(x1))))
    index = np.vectorize(temp_dict.get)(x2)
    perm = np.argsort(index)

    assert((x2[perm]==x1).all())

    return(perm)    

############################################
#       DATA GENERATION FUNCTIONS
############################################

def my_syn_data(num_timepoints=100, time=4*math.pi, num_batch=10, include_time=True, **kwargs):
    """
    
    """
    # First initialize repositories of all the data we're generating
    data_out, targets = [], []
    
    # Scaling factor for distribution mean in each class
    means = [[0.1, 1.6, 0.5],[-0.1, -0.4,-1.5]]  
    
    # Initialize the common time vector for all sequences...
    t = torch.linspace(0., time, num_timepoints)
    for __ in range(num_batch):

        # Sample the type of data based on class...
        target = np.random.choice(2, p=[.5, .5]) # Choosing the class
        
        mean1, mean2, mean3 = means[target]  # Collect the means for each signal based on the class
        
        # Sample feat 3 frequency shift
        rand_freq = 2.0*torch.rand(1)
        
        # Collect random starting places
        start = torch.rand(2) * 2 * math.pi  # Collect random starting places 
        per_t = t + 0.01*torch.randn_like(t)  # Perturb the timing so it's not the same time array for each sequence...
        feat2 = (torch.sin(start[0] + per_t) / (1 + 0.5 * per_t)).unsqueeze(-1) + mean2  # Flip the x_pos tensor to be time x feats
        feat3 = (torch.sin(start[1] + rand_freq * per_t) / (1 + 0.5 * per_t)).unsqueeze(-1)  + mean3
        
        # Construct feature1 based on the class ('corr_idx')
        if target: # if target == 1
            feat1 = (torch.cos(start[1] + rand_freq * per_t) / (1 + 0.5 * per_t)).unsqueeze(-1) + mean1
        else:
            feat1 = (torch.cos(start[0] + per_t) / (1 + 0.5 * per_t)).unsqueeze(-1) + mean1

        # Add a little random noise to the observed features
        feat1 += 0.01 * torch.randn_like(feat1)
        feat2 += 0.01 * torch.randn_like(feat2)
        feat3 += 0.01 * torch.randn_like(feat3)

        data_return = torch.concat([feat1, feat2, feat3], dim=-1)
        
        ######################
        # Easy to forget gotcha if using Neural CDEs: time should be included as a channel; CDEs need to be explicitly told 
        # the rate at which time passes. Here, we have a regularly sampled dataset, so appending time is pretty simple.
        ######################
        if include_time:
            X = torch.concat([per_t.unsqueeze(-1), data_return], dim=-1)  # Add batch dimension...
        else:
            X = data_return.clone()


        data_out.append(X)
        targets.append(target)

    data_batch = torch.stack(data_out, dim=0)
    targets = torch.from_numpy(np.array(targets))
    ######################
    # data_batch is a tensor of observations, of shape (num_batch, num_timepoints, channels=3)
    ######################

    return data_batch, targets

def syn_data_sample(T, dt, N, num_features=3, sample_rate=2, dual_sample_rate=0.5, full=False, seed=432, init_time=0):
    """
    
    """
    np.random.seed(seed)
    N_t = int(T//dt)
    y_vec, targets = my_syn_data(N_t, time=T, num_batch=N, include_time=False)

    value_cols = [f"Value_{ii+1}" for ii in range(num_features)]
    mask_cols = [f"Mask_{ii+1}" for ii in range(num_features)]
    col=["ID", "Time"] + value_cols + mask_cols + ["Cov", "Target"]
    df = pd.DataFrame(columns=col)

    for i in range(N):
        variability_num_samples = 0.5  # Variabiliity in the number of samples for each trajectory
        if variability_num_samples*2*sample_rate*T<1:
            num_samples = int(sample_rate*T)
        else:
            num_samples = np.random.randint(sample_rate*T*(1-variability_num_samples), sample_rate*T*(1+variability_num_samples))
        
        if full:
            sample_times = np.arange(N_t)
            sample_type = np.zeros((N_t, num_features)).astype(int)
        else:
            sample_times = np.random.choice(N_t, num_samples, replace=False)
            # Create a sampling mask that removes a number of features...
            sample_type = np.random.random((num_samples, num_features)) >= (1. - dual_sample_rate)
        # Pull out the sampled times from each sequence
        samples = y_vec[i, sample_times, :]

        # Non observed samples are set to 0
        samples[torch.from_numpy(sample_type)] = 0

        # Remove the times wehre all the features have been removed and update 'num_samples'
        rmv_idx = ~(samples==0).all(dim=-1)
        sample_times = sample_times[rmv_idx]
        samples = samples[rmv_idx]
        num_samples = samples.shape[0]

        # Observed samples have mask 1, others have 0.
        mask = (samples != 0).type(torch.float32)

        # Initialize a static covariate
        covs = np.zeros((num_samples, 1))

        # Package the data into the output dataframe
        individual_data = pd.DataFrame(np.concatenate((i*np.ones((num_samples, 1)), dt*np.expand_dims(sample_times, 1), samples, mask, covs, targets[i]*np.ones((num_samples, 1))), 1), columns=col)
        df = pd.concat([df, individual_data], axis=0, join='outer', ignore_index=True)
    
    df.reset_index(drop=True, inplace=True)

    return(df, y_vec)

def generate_syn_ODEData(T = 4*math.pi, dt=0.05, sample_rate=2, dual_sample_rate=0.2, num_sequences=10000, output_dir = "./"):
    """Data generating function for ODEDataset.

    --------------------------------------------
    Notes:
    """
    # Generate data with missingness
    df, full_data = syn_data_sample(T, dt, num_sequences, sample_rate=sample_rate, dual_sample_rate=dual_sample_rate)
    # Save the data to csv
    fname = f"syn_data_numSeq{num_sequences}.csv"
    df.to_csv(os.path.join(output_dir, fname), index=False)
    
    # Plot and store some examples
    N_examples = 10
    examples_dir = "syn_data_examples/"
    if not os.path.exists(os.path.join(output_dir, examples_dir)):
        os.makedirs(os.path.join(output_dir, examples_dir))
    for ex in range(N_examples):
        idx = np.random.randint(low=0, high=df["ID"].nunique())
        random_sample = df.loc[df["ID"]==idx].sort_values(by="Time").values
        rnd_full_data = full_data[idx,...]
        plt.figure()
        for dim in range(3):
            obs_mask = random_sample[:, 5+dim] == 1
            plt.plot(np.linspace(0,T,int(T//dt)), rnd_full_data[:,dim], '--', lw=1)
            plt.scatter(random_sample[obs_mask, 1], random_sample[obs_mask, 2+dim], s=30)
        plt.title("Example of a generated trajectory")
        plt.xlabel(f"Time, Target: {random_sample[0, -1]}")
        plt_name = f"syn_data_{ex}.png"
        plt.savefig(os.path.join(output_dir, examples_dir, plt_name))
        plt.close()


##########################################################
#    ODE DATA PREPROCESSING FUNCTIONS AND WRAPPERS
##########################################################

def clf_collate_fn(batch):
    """Collating batch information from CLF_Dataset"""

    pat_idx = [b['idx'] for b in batch]
    df = pd.concat([b["path"] for b in batch], axis=0)

    df_cov    = torch.Tensor(np.vstack([b["cov"] for b in batch]))

    labels  = torch.tensor(np.vstack([b["y"] for b in batch]))

    times, counts = np.unique(df.Time.values, return_counts=True)

    value_cols = [c.startswith("Value") for c in df.columns]
    mask_cols  = [c.startswith("Mask") for c in df.columns]

    batch_size = len(batch)
    num_feats = sum(value_cols)

    if batch[0]['val_samples'] is not None:
        df_after = pd.concat(b["val_samples"] for b in batch)
        df_after.sort_values(by=["ID","Time"], inplace=True)
        value_cols_val = [c.startswith("Value") for c in df_after.columns]
        mask_cols_val  = [c.startswith("Mask") for c in df_after.columns]
        X_val = torch.tensor(df_after.iloc[:,value_cols_val].values).reshape((batch_size, -1, num_feats))
        M_val = torch.tensor(df_after.iloc[:,mask_cols_val].values).reshape((batch_size, -1, num_feats))

        # Last observation before the T_val cut_off. THIS IS LIKELY TO GIVE ERRORS IF THE NUMBER OF VALIDATION SAMPLES IS HIGHER THAN 2. CHECK THIS.
        tens_last = 0
    else:
        X_val = None
        M_val = None
        tens_last = None

    res = {}
    res["pat_idx"]  = pat_idx
    res["times"]    = times
    res["X"]        = torch.tensor(df.iloc[:, value_cols].values).reshape((batch_size, -1, num_feats))
    res["M"]        = torch.tensor(df.iloc[:, mask_cols].values).reshape((batch_size, -1, num_feats))
    res["y"]        = labels
    res["cov"]      = df_cov
    res["X_val"]    = X_val
    res["M_val"]    = M_val
    res["X_last"]   = tens_last

    return res

def custom_collate_fn(batch):
    """Collating batch information from ODE_Dataset"""
    idx2batch = pd.Series(np.arange(len(batch)), index = [b["idx"] for b in batch])

    pat_idx   = [b["idx"] for b in batch]
    df        = pd.concat([b["path"] for b in batch],axis=0)
    df.sort_values(by=["Time"], inplace=True)

    # gather the mean and std of the population and feed it back to the batch
    pop_mean = batch[0]['pop_mean']
    pop_std = batch[0]['pop_std']

    df_cov    = torch.Tensor(np.vstack([b["cov"] for b in batch]))

    labels  = torch.tensor(np.vstack([b["y"] for b in batch]))

    batch_ids     = idx2batch[df.index.values].values

    ## calculating number of events at every time
    times, counts = np.unique(df.Time.values, return_counts=True)
    time_ptr      = np.concatenate([[0], np.cumsum(counts)])

    ## tensors for the data in the batch
    value_cols = [c for c in df.columns if c.startswith("Value")]
    mask_cols  = [c for c in df.columns if c.startswith("Mask")]


    if batch[0]['val_samples'] is not None:
        df_after = pd.concat(b["val_samples"] for b in batch)
        df_after.sort_values(by=["ID","Time"], inplace=True)
        value_cols_val = [c.startswith("Value") for c in df_after.columns]
        mask_cols_val  = [c.startswith("Mask") for c in df_after.columns]
        X_val = torch.tensor(df_after.iloc[:,value_cols_val].values)
        M_val = torch.tensor(df_after.iloc[:,mask_cols_val].values)
        times_val = df_after["Time"].values
        index_val = idx2batch[df_after["ID"].values].values

        X_interp = None
        M_interp = None
        times_interp = None
        index_interp = None

        # Last observation before the T_val cut_off. THIS IS LIKELY TO GIVE ERRORS IF THE NUMBER OF VALIDATION SAMPLES IS HIGHER THAN 2. CHECK THIS.
        if batch[0]["store_last"]:
            df_last = df[~df.index.duplicated(keep="last")].copy()
            index_last = idx2batch[df_last.index.values].values
        
            perm_last = sort_array_on_other(index_val,index_last)
            tens_last = torch.tensor(df_last.iloc[:,value_cols].values[perm_last,:])
            index_last = index_last[perm_last]
        else:
            index_last = 0
            tens_last = 0
    
    elif batch[0]['cal_interp'] is not None:
        df_interp = pd.concat(b['cal_interp'] for b in batch)
        X_interp = torch.tensor(df_interp[value_cols].values)
        M_interp = torch.tensor(df_interp[mask_cols].values)
        times_interp = df_interp['Time'].values
        index_interp = idx2batch[df_interp['ID'].values].values
        
        df_extrap = pd.concat(b['cal_extrap'] for b in batch)
        X_val = torch.tensor(df_extrap[value_cols].values)
        M_val = torch.tensor(df_extrap[mask_cols].values)
        times_val = df_extrap['Time'].values
        index_val = idx2batch[df_extrap["ID"].values].values

        tens_last = None
        index_last = None
    
    else:
        X_interp = None
        M_interp = None
        times_interp = None
        index_interp = None
        X_val = None
        M_val = None
        times_val = None
        index_val = None
        tens_last = None
        index_last = None


    res = {}
    res["pat_idx"]  = pat_idx
    res["times"]    = times
    res["time_ptr"] = time_ptr
    res["X"]        = torch.tensor(df[value_cols].values)
    res["M"]        = torch.tensor(df[mask_cols].values)
    res["obs_idx"]  = torch.tensor(batch_ids)
    res["y"]        = labels
    res["cov"]      = df_cov
    res["X_interp"] = X_interp
    res["M_interp"] = M_interp
    res["times_interp"] = times_interp
    res["index_interp"] = index_interp
    res["X_val"]    = X_val
    res["M_val"]    = M_val
    res["times_val"]= times_val
    res["index_val"]= index_val
    res["X_last"]   = tens_last
    res["obs_idx_last"]= index_last
    res['pop_mean'] = pop_mean
    res['pop_std'] = pop_std

    return res

class ODE_Dataset(Dataset):
    """
    Dataset class for ODE type of data (index based representation of sequences (time x Values x Masks))
    Can be fed with either a csv file containg the dataframe or directly with a panda dataframe.
    One can further provide samples idx that will be used (for training / validation split purposes.)
    """
    def __init__(self, csv_file=None, cov_file=None, label_file=None, 
                panda_df=None, cov_df=None, label_df=None, root_dir="./", 
                t_mult=1.0, idx=None, validation=False, val_options=None, T_val=None, calibration_test=False, 
                add_noise=None, provide_pop_stats=False, pop_mean=None, pop_std=None, dataset_name='syn_data'):
        """
        Args:
            csv_file      CSV file to load the dataset from
            cov_file      path to pre-processed CSV of time series covariates
            label_file    path to pre-processed CSV of time series labels
            panda_df      alternatively use pandas df instead of CSV file
            cov_df        alternative pandas df instead of CSV file
            label_df      alternative pandas df instead of CSV file
            root_dir      directory of the CSV file(s)
            t_mult        a scalar multiplier of the times (optional)
            idx           subset of indices of the dataset to use (helpful for train/val/test splitting)
            validation    whether the dataset is for validation purposes
            val_options   dictionary with validation dataset options.
                                    T_val : Time after which observations are considered as test samples
                                    max_val_samples : maximum number of test observations per trajectory.
            T_val         Time after which observations are considered as test samples
            calibration_test  Whether we'll extract a test dataset for calibration purposes
            add_noise     whether to add temporally expanding noise to the time series
            provide_pop_stats whether to provide population mean and std.dev
            pop_mean      If provided, used as the population mean of the dataset
            pop_std       If provided, used as the population std.dev of the dataset
            delta_t       minimum time step between possible observations
            dataset_name  the string identifier of the dataset we are processing

        """
        self.validation = validation
        self.T_val = T_val
        self.calibration_test = calibration_test

        if panda_df is not None:
            assert (csv_file is None), "Only one feeding option should be provided, not both"
            self.df = panda_df
            self.cov_df = cov_df
            self.label_df  = label_df
        else:
            assert (csv_file is not None) , "At least one feeding option required !"
            self.df = pd.read_csv(root_dir + "/" + csv_file)
            assert self.df.columns[0]=="ID"
            if label_file is None:
                self.label_df = None
            else:
                self.label_df = pd.read_csv(root_dir + "/" + label_file)
                assert self.label_df.columns[0]=="ID"
                assert self.label_df.columns[1]=="label"
            if cov_file is None :
                self.cov_df = None
            else:
                self.cov_df = pd.read_csv(root_dir + "/" + cov_file)
                assert self.cov_df.columns[0]=="ID"

        #Create Dummy covariates and labels if they are not fed.
        if self.cov_df is None:
            num_unique = np.zeros(self.df["ID"].nunique())
            self.cov_df = pd.DataFrame({"ID":self.df["ID"].unique(),"Cov": num_unique})
        if self.label_df is None:
            if 'Target' in self.df.columns:
                self.label_df = self.df.groupby("ID", group_keys=False)['Target'].agg('first')
                self.label_df = self.label_df.reset_index().rename(columns={'Target': 'label'})
            else:  # Need to figure out what to do best in this situation where we don't have labels provided....
                num_unique = np.zeros(self.df["ID"].nunique())
                self.label_df = pd.DataFrame({"ID":self.df["ID"].unique(),"label": num_unique})

        #If validation : consider only the data with a least one observation before T_val and one observation after:
        self.store_last = False

        if self.validation:
            df_beforeIdx = self.df.loc[self.df["Time"]<=val_options["T_val"],"ID"].unique()
            if val_options.get("T_val_from"): #Validation samples only after some time.
                df_afterIdx  = self.df.loc[self.df["Time"]>=val_options["T_val_from"],"ID"].unique()
                self.store_last = True #Dataset get will return a flag for the collate to compute the last sample before T_val
            else:
                df_afterIdx  = self.df.loc[self.df["Time"]>val_options["T_val"],"ID"].unique()
            
            valid_idx = np.intersect1d(df_beforeIdx,df_afterIdx)
            self.df = self.df.loc[self.df["ID"].isin(valid_idx)]
            self.label_df = self.label_df.loc[self.label_df["ID"].isin(valid_idx)]
            self.cov_df   = self.cov_df.loc[self.cov_df["ID"].isin(valid_idx)]

        # Create a subset of the data with the specified list of indices
        if idx is not None:
            self.df = self.df.loc[self.df["ID"].isin(idx)].copy()
            map_dict= dict(zip(self.df["ID"].unique(),np.arange(self.df["ID"].nunique())))
            self.df["ID"] = self.df["ID"].map(map_dict) # Reset the ID index.

            self.cov_df = self.cov_df.loc[self.cov_df["ID"].isin(idx)].copy()
            self.cov_df["ID"] = self.cov_df["ID"].map(map_dict) # Reset the ID index.

            self.label_df = self.label_df.loc[self.label_df["ID"].isin(idx)].copy()
            self.label_df["ID"] = self.label_df["ID"].map(map_dict) # Reset the ID index.

        assert self.cov_df.shape[0]==self.df["ID"].nunique()

        
        if self.calibration_test:
            # Extract the IDs of the dataframe that have entries both before and after the Validation Time point...
            df_beforeIdx = self.df.loc[self.df["Time"] <= self.T_val, "ID"].unique()
            df_afterIdx  = self.df.loc[self.df["Time"] > self.T_val, "ID"].unique()
            valid_idx = np.intersect1d(df_beforeIdx,df_afterIdx)
            # Constrain the dataframe to only these indices
            self.df = self.df.loc[self.df["ID"].isin(valid_idx)]
            
            # Create an interpolation and extrapolation dataframe, we'll use the interpolation dataframe
            #  to sample rows of observations from for held out calibration analysis. We'll use the entire
            #  extrapoloation dataframe for this same analysis...
            self.df_interp = self.df.loc[self.df.Time <= self.T_val]
            self.df_extrap = self.df.loc[self.df.Time > self.T_val]

            # Group the interpolation DataFrame by ID and then sample rows
            self.df_interp_sampled = self.df_interp.groupby("ID").sample(frac=0.15)

            # Remove the sampled rows from the base interpolation DataFrame
            self.df_interp = self.df_interp[~self.df_interp.index.isin(self.df_interp_sampled.index)]

            # Account for Sequences that we may not have kept after the sampling (too few observations before T_val)
            # Correct across all dataframes... Rename df_interp to df because this is the main dataframe that 
            #      we'll be using to predict the distributions
            self.df_interp_sampled = self.df_interp_sampled.astype(np.float32)
            self.df = self.df_interp[self.df_interp.ID.isin(self.df_interp_sampled.ID.unique())].copy().astype(np.float32)
            self.df_extrap = self.df_extrap[self.df_extrap.ID.isin(self.df_interp.ID.unique())].astype(np.float32)
            self.cov_df = self.cov_df[self.cov_df.ID.isin(self.df_interp.ID.unique())].astype(np.float32)
            self.label_df = self.label_df[self.label_df.ID.isin(self.df_interp.ID.unique())]

            # Re-index the IDs for sampling batches from the dataset...
            map_dict = dict(zip(self.df["ID"].unique(),np.arange(self.df["ID"].nunique())))
            self.df["ID"] = self.df["ID"].map(map_dict) # Reset the ID index.
            self.df_interp_sampled["ID"] = self.df_interp_sampled["ID"].map(map_dict)
            self.df_extrap["ID"] = self.df_extrap["ID"].map(map_dict)
            self.cov_df["ID"] = self.cov_df["ID"].map(map_dict) # Reset the ID index.
            self.label_df["ID"] = self.label_df["ID"].map(map_dict) # Reset the ID index.

        
        value_cols = [c for c in self.df.columns if c.startswith("Value")]
        mask_cols = [c for c in self.df.columns if c.startswith("Mask")]
        
        # Calculate the population statistics if we're not creating a validation dataset
        # This will be done only if `provide_pop_stats` is true, ideally only during training
        # If we want these provided for the validation and testing datasets, we'll expect that 
        #    the pop mean and std are provided as inputs...
        if not validation and provide_pop_stats:
            masks = self.df[mask_cols].values
            values = self.df[value_cols].values
            values[masks==0] = np.nan
            self.pop_mean = np.nanmean(values, axis=0)
            self.pop_std = np.nanstd(values, axis=0)
        else: # Fill with the provided values or None
            self.pop_mean = pop_mean
            self.pop_std = pop_std


        self.variable_num = len(value_cols) #number of variables in the dataset
        self.cov_dim = self.cov_df.shape[1]-1

        self.cov_df = self.cov_df.astype(np.float32)
        self.cov_df.set_index("ID", inplace=True)

        self.label_df.set_index("ID",inplace=True)

        self.df.Time    = self.df.Time * t_mult

        # Add noise to the observations if add_noise is not None
        # Primarily in place for the Test data... 
        if add_noise is not None:
            if dataset_name == 'mimic_extract':
                noise_levels = np.linspace(0, 0.5, 10)
            elif dataset_name == 'physionet':
                noise_levels = np.linspace(0, 0.5, 10)
            elif dataset_name == 'activity':
                noise_levels = np.linspace(0, 250, 10)
            elif dataset_name == 'gestures':
                noise_levels = np.linspace(0, 6, 10)
            else:
                noise_levels = np.linspace(0, 0.7, 10)

            noise_rate = noise_levels[int(add_noise)]
            base_noise, rate = 0.1, (1+noise_rate)
            time_dep_scale = base_noise*(rate**self.df.Time)
            # Generate time dependent noise for each column based on the time of each row :)
            time_dep_noise = np.random.normal(scale=time_dep_scale, size=(self.variable_num, time_dep_scale.shape[0])) 

            # Loop through each value column and add the noise. We conveniently mask out the unobserved entries so no worries about adding to the "empty" entries.
            for i, c in enumerate(value_cols):
                self.df[c] = self.df[c] + time_dep_noise[i, :]

        
        # Ensure that all data is float32
        self.df = self.df.astype(np.float32)


        if self.validation:
            assert val_options is not None, "Validation set options should be fed"
            self.df_before = self.df.loc[self.df["Time"]<=val_options["T_val"]].copy()
            if val_options.get("T_val_from"): #Validation samples only after some time.
                self.df_after  = self.df.loc[self.df["Time"]>=val_options["T_val_from"]].sort_values("Time").copy()
            else:
                self.df_after  = self.df.loc[self.df["Time"]>val_options["T_val"]].sort_values("Time").copy()

            if val_options.get("T_closest") is not None:
                df_after_temp = self.df_after.copy()
                df_after_temp["Time_from_target"] = (df_after_temp["Time"]-val_options["T_closest"]).abs()
                df_after_temp.sort_values(by=["Time_from_target","Value_0"], inplace = True,ascending=True)
                df_after_temp.drop_duplicates(subset=["ID"],keep="first",inplace = True)
                self.df_after = df_after_temp.drop(columns = ["Time_from_target"])
            else:
                self.df_after  = self.df_after.groupby("ID", group_keys=False).head(val_options["max_val_samples"]).copy()

            self.df = self.df_before #We remove observations after T_val


            self.df_after.ID = self.df_after.ID.astype(int)
            self.df_after.sort_values("Time", inplace=True)
        else:
            self.df_after = None


        self.length     = self.df["ID"].nunique()
        self.df.ID      = self.df.ID.astype(int)
        self.df.set_index("ID", inplace=True)

        self.df.sort_values("Time", inplace=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        subset = self.df.loc[idx]
        if len(subset.shape)==1: #Don't ask me anything about this (Note from EdB).
            subset = self.df.loc[[idx]]

        covs = self.cov_df.loc[idx].values
        tag  = self.label_df.loc[idx].astype(np.float32).values
        if self.validation :
            val_samples = self.df_after.loc[self.df_after["ID"]==idx]
        else:
            val_samples = None

        if self.calibration_test:
            cal_interp = self.df_interp_sampled.loc[self.df_interp_sampled["ID"] == idx]
            cal_extrap = self.df_extrap.loc[self.df_extrap["ID"] == idx]
        else:
            cal_interp = None
            cal_extrap = None

        ## returning also idx to allow empty samples
        return {"idx":idx, "y": tag, "path": subset, "cov": covs , "val_samples":val_samples, "cal_interp": cal_interp, "cal_extrap": cal_extrap, "store_last":self.store_last, "pop_mean": self.pop_mean, "pop_std": self.pop_std}

class CLF_Dataset(Dataset):
    """
    Dataset class for CLF type of data (tensor representation of sequences (time x Values + Masks))
    Can be fed with either a csv file containg the dataframe or directly with a panda dataframe.
    One can further provide samples idx that will be used (for training / validation split purposes.)
    """
    
    def _grouped_expand(self, group, times):
        """
        Expand to provide a row for all possible times for each timeseries. 
        This helps to fill in the missing values of the timeseries with NaN
        Here, `group` is a grouped dataframe for each timeseries ID.
        """
        group = group.set_index('Time')
        group = group.reindex(times)
        # Fill forward and backward administrative information
        group.loc[:, ['ID']] = group.loc[:, ['ID']].ffill().bfill()
        # Reset index and return the expanded group
        group = group.reset_index()
        
        return group
    
    def _expand_columns(self, df, value_cols, mask_cols, dt, st, et, integer_index=False):
        """
        Takes the unordered time series data and expands it to be uniform length, filling missing values with NaN
        """
        # For the expansion, we'll just create a integer index of the time steps (makes reindexing much smoother)
        all_times = np.arange(int(et//dt))
        if integer_index:
            df['Time'] = df['Time']//dt
        else:
            df['Time'] = df['Time']/dt  # Will convert back into "time" after the column expansion
        
        # First remove the values and masks from the dataframe and set all non-observed values to NaN
        values = df[value_cols].values
        masks = df[mask_cols].values
        
        values[masks==0] = np.nan
        
        # Replace the values back in the dataframe and delete the mask columns
        df[value_cols] = values
        df = df.drop(columns=mask_cols)  # We'll regenerate the mask later
        
        df_exp = df.groupby('ID').apply(lambda x: self._grouped_expand(x, all_times)).reset_index(drop=True)

        # Create the new mask columns
        vitals = df_exp.set_index(['ID', 'Time'])  # isolate indexing columns
        obs_mask = vitals.notnull().astype(int)
        
        mask_names = [(c,"Mask_"+c.split('_')[-1]) for c in obs_mask.columns]
        obs_mask.rename(columns = dict(mask_names), inplace=True)
        
        # Now combine to produce the final expanded dataframe
        df = pd.concat([vitals, obs_mask], axis=1).fillna(0.0).reset_index()

        df['Time'] = df['Time']*dt  # Converting back to "time"
        
        
        return df
        
    
    def __init__(self, 
                    csv_file=None, cov_file=None, label_file=None, panda_df=None, 
                    cov_df=None, label_df=None, root_dir="./", delta_t=0.05, 
                    start_time=0.0, end_time=5.0, idx=None, validation=False, 
                    val_options=None, add_noise=None, integer_index=False, dataset_name='syn_data'):
        """
        Args:
            csv_file      CSV file to load the dataset from
            cov_file      path to pre-processed CSV of time series covariates
            label_file    path to pre-processed CSV of time series labels
            panda_df      alternatively use pandas df instead of CSV file
            cov_df        alternative pandas df instead of CSV file
            label_df      alternative pandas df instead of CSV file
            root_dir      directory of the CSV file(s)
            delta_t       minimum time step between possible observations
            start_time    the assumed start time of the time series
            end_time      maximum observation time of the time series
            idx           subset of indices of the dataset to use (helpful for train/val/test splitting)
            validation    whether the dataset is for validation purposes
            val_options  dictionnary with validation dataset options.
                                    T_val : Time after which observations are considered as test samples
                                    max_val_samples : maximum number of test observations per trajectory.
            add_noise     whether to add temporally expanding noise to the time series
            integer_index whether the index in incremented
            dataset_name  the string identifier of the dataset we are processing
        """
        self.validation = validation

        if panda_df is not None:
            assert (csv_file is None), "Only one feeding option should be provided, not both"
            self.df = panda_df
            self.cov_df = cov_df
            self.label_df  = label_df
        else:
            assert (csv_file is not None) , "At least one feeding option required !"
            self.df = pd.read_csv(root_dir + "/" + csv_file)
            assert self.df.columns[0]=="ID"
            if label_file is None:
                self.label_df = None
            else:
                self.label_df = pd.read_csv(root_dir + "/" + label_file)
                assert self.label_df.columns[0]=="ID"
                assert self.label_df.columns[1]=="label"
            if cov_file is None :
                self.cov_df = None
            else:
                self.cov_df = pd.read_csv(root_dir + "/" + cov_file)
                assert self.cov_df.columns[0]=="ID"

        #Create dummy covariates and labels if they are not fed.
        if self.cov_df is None:
            num_unique = np.zeros(self.df["ID"].nunique())
            self.cov_df = pd.DataFrame({"ID":self.df["ID"].unique(),"Cov": num_unique})
            if 'Cov' in self.df.columns:  # Delete 'Cov' from the df
                self.df = self.df.drop(columns=['Cov'])
        if self.label_df is None:
            if 'Target' in self.df.columns:
                self.label_df = self.df.groupby("ID", group_keys=False)['Target'].agg('first')
                self.label_df = self.label_df.reset_index().rename(columns={'Target': 'label'})
                # Remove 'Target' from self.df
                self.df = self.df.drop(columns=['Target'])
            else:  # Need to figure out what to do best in this situation where we don't have labels provided....
                num_unique = np.zeros(self.df["ID"].nunique())
                self.label_df = pd.DataFrame({"ID":self.df["ID"].unique(),"label": num_unique})

        #If validation : consider only the data with a least one observation before T_val and one observation after:
        self.store_last = False
        
        value_cols = [c for c in self.df.columns if c.startswith("Value")]
        mask_cols = [c for c in self.df.columns if c.startswith("Mask")]

        # Expand the dataframe over the temporal columns
        self.df = self._expand_columns(self.df, value_cols, mask_cols, delta_t, start_time, end_time, integer_index)        
        
        if self.validation:
            df_beforeIdx = self.df.loc[self.df["Time"]<=val_options["T_val"],"ID"].unique()
            if val_options.get("T_val_from"): #Validation samples only after some time.
                df_afterIdx  = self.df.loc[self.df["Time"]>=val_options["T_val_from"],"ID"].unique()
                self.store_last = True #Dataset get will return a flag for the collate to compute the last sample before T_val
            else:
                df_afterIdx  = self.df.loc[self.df["Time"]>val_options["T_val"],"ID"].unique()
            
            valid_idx = np.intersect1d(df_beforeIdx,df_afterIdx)
            self.df = self.df.loc[self.df["ID"].isin(valid_idx)]
            self.label_df = self.label_df.loc[self.label_df["ID"].isin(valid_idx)]
            self.cov_df   = self.cov_df.loc[self.cov_df["ID"].isin(valid_idx)]
        
        # Create a subset of the data with the specified list of indices
        if idx is not None:
            self.df = self.df.loc[self.df["ID"].isin(idx)].copy()
            map_dict= dict(zip(self.df["ID"].unique(),np.arange(self.df["ID"].nunique())))
            self.df["ID"] = self.df["ID"].map(map_dict) # Reset the ID index.

            self.cov_df = self.cov_df.loc[self.cov_df["ID"].isin(idx)].copy()
            self.cov_df["ID"] = self.cov_df["ID"].map(map_dict) # Reset the ID index.

            self.label_df = self.label_df.loc[self.label_df["ID"].isin(idx)].copy()
            self.label_df["ID"] = self.label_df["ID"].map(map_dict) # Reset the ID index.

        assert self.cov_df.shape[0]==self.df["ID"].nunique()        
        
        self.variable_num = len(value_cols) #number of variables in the dataset
        self.cov_dim = self.cov_df.shape[1]-1

        self.cov_df = self.cov_df.astype(np.float32)
        self.cov_df.set_index("ID", inplace=True)

        self.label_df.set_index("ID",inplace=True)

        # Add noise to the observations if add_noise is not None
        # Primarily in place for the Test data... 
        if add_noise is not None:
            if dataset_name == 'mimic_extract':
                noise_levels = np.linspace(0, 0.75, 10)
            elif dataset_name == 'physionet':
                noise_levels = np.linspace(0, 1.0, 10)
            elif dataset_name == 'activity':
                noise_levels = np.linspace(0, 500, 10)
            elif dataset_name == 'gestures':
                noise_levels = np.linspace(0, 6, 10)
            else:
                noise_levels = np.linspace(0, 0.75, 10)

            noise_rate = noise_levels[int(add_noise)]
            base_noise, rate = 0.1, (1+noise_rate)
            time_dep_scale = base_noise*(rate**self.df.Time)
            # Generate time dependent noise for each column based on the time of each row :)
            time_dep_noise = np.random.normal(scale=time_dep_scale, size=(self.variable_num, time_dep_scale.shape[0])) 

            # Loop through each value column and add the noise. We conveniently mask out the unobserved entries so no worries about adding to the "empty" entries.
            for i, c in enumerate(value_cols):
                self.df[c] = self.df[c] + time_dep_noise[i, :]

        
        # Ensure that all data is float32
        self.df = self.df.astype(np.float32)


        if self.validation:
            assert val_options is not None, "Validation set options should be fed"
            self.df_before = self.df.loc[self.df["Time"]<=val_options["T_val"]].copy()
            if val_options.get("T_val_from"): #Validation samples only after some time.
                self.df_after  = self.df.loc[self.df["Time"]>=val_options["T_val_from"]].sort_values("Time").copy()
            else:
                self.df_after  = self.df.loc[self.df["Time"]>val_options["T_val"]].sort_values("Time").copy()

            if val_options.get("T_closest") is not None:
                df_after_temp = self.df_after.copy()
                df_after_temp["Time_from_target"] = (df_after_temp["Time"]-val_options["T_closest"]).abs()
                df_after_temp.sort_values(by=["Time_from_target","Value_0"], inplace = True,ascending=True)
                df_after_temp.drop_duplicates(subset=["ID"],keep="first",inplace = True)
                self.df_after = df_after_temp.drop(columns = ["Time_from_target"])
            else:
                self.df_after  = self.df_after.groupby("ID", group_keys=False).head(val_options["max_val_samples"]).copy()

            self.df = self.df_before #We remove observations after T_val


            self.df_after.ID = self.df_after.ID.astype(int)
            self.df_after.sort_values("Time", inplace=True)
        else:
            self.df_after = None


        self.length     = self.df["ID"].nunique()
        self.df.ID      = self.df.ID.astype(int)
        self.df.set_index("ID", inplace=True)

        self.df.sort_values("Time", inplace=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        subset = self.df.loc[idx]
        if len(subset.shape)==1: #Don't ask me anything about this (Note from EdB).
            subset = self.df.loc[[idx]]

        covs = self.cov_df.loc[idx].values
        tag  = self.label_df.loc[idx].astype(np.float32).values
        if self.validation :
            val_samples = self.df_after.loc[self.df_after["ID"]==idx]
        else:
            val_samples = None

        ## returning also idx to allow empty samples
        return {"idx":idx, "y": tag, "path": subset, "cov": covs , "val_samples":val_samples, "store_last":self.store_last}

def create_clf_dataset(params, add_noise=None, device='cpu'):
    """
    Using the provided `params`, construct the dataset for use in a Prediction model for classification.
    """
    
    # Gather all of the basic dataset parameters and configurations...
    syn_data = params.get('syn_data', True)  # Whether the data is synthetic or derived from a csv file.
    dataset_name = params.get('dataset_name', 'syn_data')  # The type of dataset we'll be using
    seed = params.get('seed', 2022)
    N = params.get('num_sequences', 10000)
    
    # The filename of the dataset we'll be using (all data has been pre-processed and stored in a .csv file)
    dataset_dir = params.get('dataset_dir', f"datasets/{dataset_name}/{dataset_name}_numSeq{N}.csv")  
    if dataset_name  == 'syn_data':
        sample_rate = params.get('sample_rate', 2)
        dual_sample_rate = params.get('dual_sample_rate', 0.2)
        t_val = params.get("max_time_val", 3*math.pi)
        val_samples = params.get("max_val_samples", 1)
    else:
        raise ValueError("Dataset name is not recognized")

    delta_t = params.get('delta_t', 0.05)
    T = params.get('max_time', 4*math.pi)
    tr_batch_size = params.get('tr_batch_size', 500)
    
    # Check whether the dataset file has been created previously or not
    # If not, create it.
    if not os.path.isfile(dataset_dir):
        
        if dataset_name == 'syn_data':
            generate_syn_ODEData(T=T, det=delta_t, sample_rate=sample_rate, duale_sample_rate=dual_sample_rate, output_dir= f"./datasets/{dataset_name}/")
        else:
            raise ValueError("Dataset name is not recognized")

    if dataset_name in ['syn_data', 'gestures', 'activity']:
        cov_file = None
        label_file = None
    else:
        raise ValueError("Dataset name is not recognized")
    
    # Split the dataset into a train/val set (at least by index)
    if dataset_name in ['syn_data']:
        train_idx, test_idx = train_test_split(np.arange(N), test_size=0.2, random_state=seed)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=seed)

    # Create the training and validation datasets
    integer_index = True if dataset_name == 'physionet' else False

    data_train = CLF_Dataset(csv_file=dataset_dir, cov_file=cov_file, label_file=label_file, idx=train_idx,  
                             delta_t=delta_t, end_time=T, integer_index=integer_index, dataset_name=dataset_name)
    val_options = {"T_val": t_val, "max_val_samples": val_samples}
    data_val = CLF_Dataset(csv_file=dataset_dir, cov_file=cov_file, label_file=label_file, idx=val_idx, val_options=val_options, 
                            delta_t=delta_t, end_time=T, integer_index=integer_index, dataset_name=dataset_name)
    data_test = CLF_Dataset(csv_file=dataset_dir, cov_file=cov_file, label_file=label_file, idx=test_idx, val_options=val_options, 
                            add_noise=add_noise,  delta_t=delta_t, end_time=T, integer_index=integer_index, dataset_name=dataset_name)

    # Create the dataloaders
    train_loader = DataLoader(dataset=data_train, collate_fn=clf_collate_fn, shuffle=True, batch_size=tr_batch_size, num_workers=2)
    val_loader = DataLoader(dataset=data_val, collate_fn=clf_collate_fn, shuffle=False, batch_size=len(data_val), num_workers=1)
    test_loader = DataLoader(dataset=data_test, collate_fn=clf_collate_fn, shuffle=False, batch_size=len(data_test), num_workers=1)

    return train_loader, val_loader, test_loader, params

def create_ode_dataset(params, constrain_test_size=False, add_noise=None, device='cpu'):
    """Using the provided parameters, construct the dataset for use in a NODE Model"""
    
    # Gather all of the basic dataset parameters and configurations...
    dataset_name = params.get('dataset_name', 'syn_data')  # The type of dataset we'll be using
    seed = params.get('seed', 2022)
    N = params.get('num_sequences', 10000)
    
    # The filename of the dataset we'll be using (all data has been pre-processed and stored in a .csv file)
    dataset_dir = params.get('dataset_dir', f"datasets/{dataset_name}/{dataset_name}_numSeq{N}.csv")  
    if dataset_name == 'syn_data':
        sample_rate = params.get('sample_rate', 2)
        dual_sample_rate = params.get('dual_sample_rate', 0.2)
        t_val = params.get("max_time_val", 3*math.pi)
        val_samples = params.get("max_val_samples", 1)
    else:
        raise ValueError("Dataset name is not recognized")


    delta_t = params.get('delta_t', 0.05)
    T = params.get('max_time', 4*math.pi)
    tr_batch_size = params.get('tr_batch_size', 500)
    
    # Check whether the dataset file has been created previously or not
    # If not, create it.
    if not os.path.isfile(dataset_dir):
        
        if dataset_name == 'syn_data':
            generate_syn_ODEData(T=T, det=delta_t, sample_rate=sample_rate, duale_sample_rate=dual_sample_rate, output_dir= f"./datasets/{dataset_name}/")
        else:
            raise ValueError("Dataset name is not recognized")

    if dataset_name == 'syn_data':
        cov_file = None
        label_file = None
    else:
        raise ValueError("Dataset name is not recognized")
    
    # Split the dataset into a train/val set (at least by index)
    if dataset_name in ['syn_data']:
        train_idx, test_idx = train_test_split(np.arange(N), test_size=0.2, random_state=seed)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=seed)

    # Create the training and validation datasets
    data_train = ODE_Dataset(csv_file=dataset_dir, cov_file=cov_file, label_file=label_file, 
                             idx=train_idx, provide_pop_stats=True, dataset_name=dataset_name)
    val_options = {"T_val": t_val, "max_val_samples": val_samples}
    data_val = ODE_Dataset(csv_file=dataset_dir, cov_file=cov_file, label_file=label_file, idx=val_idx, val_options=val_options, 
                           validation=True, pop_mean=data_train.pop_mean, pop_std=data_train.pop_std, dataset_name=dataset_name)
    data_test = ODE_Dataset(csv_file=dataset_dir, cov_file=cov_file, label_file=label_file, idx=test_idx, val_options=val_options, 
                            validation=True, add_noise=add_noise, pop_mean=data_train.pop_mean, pop_std=data_train.pop_std, dataset_name=dataset_name)

    # Create the dataloaders
    train_loader = DataLoader(dataset=data_train, collate_fn=custom_collate_fn, shuffle=True, batch_size=tr_batch_size, num_workers=2)
    val_loader = DataLoader(dataset=data_val, collate_fn=custom_collate_fn, shuffle=False, batch_size=len(data_val), num_workers=1)
    if constrain_test_size:
        test_loader = DataLoader(dataset=data_test, collate_fn=custom_collate_fn, shuffle=False, batch_size=tr_batch_size, num_workers=1)
    else:
        test_loader = DataLoader(dataset=data_test, collate_fn=custom_collate_fn, shuffle=False, batch_size=len(data_test), num_workers=1)

    return train_loader, val_loader, test_loader, params

def define_dataset(params, add_noise=None, constrain_test_size=False, device='cpu'):
    """Using the params dictionary, construct a dataset, initialize a dataloader and return to the model training script"""

    if params.dataset_format == 'ode':
        train_loader, val_loader, test_loader, params = create_ode_dataset(params, constrain_test_size, add_noise, device)
    elif params.dataset_format == 'clf':
        train_loader, val_loader, test_loader, params = create_clf_dataset(params, add_noise, device)

    return train_loader, val_loader, test_loader, params
