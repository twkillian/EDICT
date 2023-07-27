""" model_utils.py
Utility and Helper functions for use to facilitate modeling the temporal evolution of an irregular timeseries

GRU-ODE-Bayes Code is derived and modified from https://github.com/edebrouwer/gru_ode_bayes


Notes:
 - 
"""

############################################
#           IMPORTS and DEFINITIONS
############################################
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from torchdiffeq import odeint

from GRUD import GRUD_Layer
from IPN import Interpolator
from losses import log_normal_nll, niw_nll, compute_KL_loss, niw_reg_error

############################################
###   EVIDENTIAL DISTRIBUTION MODULES
############################################

class log_normal_dist(nn.Module):
    """Transforms hidden state into an implicit Log-Normal distribution over the observations"""
    def __init__(self, hidden_size, p_hidden, input_size, bias=True, dropout_rate=0):
        super(log_normal_dist, self).__init__()

        self.module = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, p_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(p_hidden, 2*input_size, bias=bias)
        )

    def forward(self, h_t):
        return self.module(h_t)


class normal_inverse_wishart(nn.Module):
    """
    Construct a NIW evidential distribution from the propagated hidden state.
    
    We're going to simplify things slightly and only construct a diagonal covariance matrix...
    But, also leverage the module idea for each element of the NIW dist rather than a single linear layer
    as I'd done before. Hopefully this will add a bit of stability....
    """
    def __init__(self, hidden_size, p_hidden, input_size, bias=True, dropout_rate=0, eps=10e-6):
        super(normal_inverse_wishart, self).__init__()

        self.output_dim = input_size
        self.eps = eps

        # Module for the prior on the mean of the NIW distribution
        self.mu_module = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, p_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(p_hidden, input_size, bias=bias)
        )
        # Module for the pseudo evidence for the mean
        self.lambda_module = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, p_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(p_hidden, 1, bias=bias),
        )
        # Module for the psuedo evidence for the covariance
        self.nu_module = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, p_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(p_hidden, 1, bias=bias),
        )
        # Module for the prior on the NIW covariance
        self.psi_module = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, p_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(p_hidden, input_size)
        )

    def evidence(self, x):
        return torch.log(torch.exp(x) + 1)
        

    def forward(self, h_t):
        out_mu = self.mu_module(h_t)
        out_lambda = self.evidence(self.lambda_module(h_t)) + 1.0  # Ensuring that we stay away from zero...
        out_nu = self.evidence(self.nu_module(h_t)) + self.output_dim + 2  # Hacky way to satisfy the constraint?
        # The constraint here (dim + 1) is due to calcuating the mean of the Inverse Wishart dist (used for computing the variance of a prediction)
        # This is in contrast to the constraint for the NIW prior being (dim - 1)

        # Only outputting the diagonal of the covariance prior Psi... (assuming they are log variances)
        out_psi = self.psi_module(h_t)

        return out_mu, out_lambda, out_psi, out_nu

############################################
###  GRU-ODE-Bayes FUNCTIONS AND MODELS
############################################
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.05)

class covariates_mapping(nn.Module):
    """Construct a mapping from the covariates to initialize the hidden state."""
    def __init__(self, cov_size, cov_hidden, hidden_size, bias=True, dropout_rate=0):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.Linear(cov_size, cov_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(cov_hidden, hidden_size, bias=bias),
            torch.nn.Tanh()
        )
    
    def forward(self, cov):
        return self.mapping(cov)

class clf_model(nn.Module):
    """Construct a simple classification module."""
    def __init__(self, hidden_size, clf_hidden, output_dims, bias=True, dropout_rate=0):
        super().__init__()
        self.clf_module = torch.nn.Sequential(
            torch.nn.Linear(hidden_size,clf_hidden,bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(clf_hidden, output_dims, bias=bias)
        )

    def forward(self, h_t):
        return self.clf_module(h_t)

class GRUObsCell(nn.Module):
    """Implements discrete update based on the received observations."""

    def __init__(self, input_size, hidden_size, prep_hidden, bias=True, dist_type='log_normal', reweighting=False, rewt=1.96, pop_mean=None, pop_std=None):
        super(GRUObsCell, self).__init__()
        self.gru_d     = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)
        self.gru_debug = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)

        ## prep layer and its initialization
        std            = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep    = torch.nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = torch.nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

        self.input_size  = input_size
        self.prep_hidden = prep_hidden
        self.dist_type = dist_type
        self.reweighting = reweighting
        self.rewt = rewt

        self.pop_mean = pop_mean
        self.pop_std = pop_std

    def forward(self, h, p, X_obs, M_obs, i_obs):

        ## only updating rows that have observations
        if self.dist_type == 'log_normal':
            p_obs        = p[i_obs]
        elif self.dist_type == 'niw':
            mean, lmbda, logvar, nu = p
            mean_obs = mean[i_obs]
            lmbda_obs = lmbda[i_obs]
            logvar_obs = logvar[i_obs]
            nu_obs = nu[i_obs]
            p_obs = (mean_obs, lmbda_obs, logvar_obs, nu_obs)

        if self.dist_type == 'log_normal':
            mean, logvar = torch.chunk(p_obs, 2, dim=1)
            sigma = torch.exp(0.5 * logvar)
        elif self.dist_type == 'niw':
            mean, lmbda, logvar, nu = p_obs
            sigma = torch.sqrt(torch.exp(logvar) / (lmbda*(nu - mean.shape[-1] - 1))) 

        # If observation is far out of distribution, we want to use more of the mean for the subsequent update...
        # We use a distance metric derived from the RBF Kernel to compute this reweighting
        if self.reweighting:
            # Version 1.0 -- Adaptively clip outliers to outer range of distribution... 
            # Currently using a static hyperparameter self.rewt to reflect that threshold
            # --> Can tune and perhaps change to be time-dependent / recurrent?
            if self.pop_mean is None:
                lower, upper = mean - self.rewt*sigma, mean + self.rewt*sigma
            else: # Evaluating against a sanity-check baseline that applying population averages aren't as effective *fingers crossed*
                lower, upper = self.pop_mean - self.rewt*self.pop_std, mean + self.rewt*self.pop_std

            X_obs = torch.clamp(X_obs, min=lower, max=upper)  # The magic of pytorch lets us handle this on a per-entry basis

        # Compute the normalized error
        error = (X_obs - mean) / sigma

        gru_input    = torch.stack([X_obs, mean, logvar, error], dim=2).unsqueeze(2)
        gru_input    = torch.matmul(gru_input, self.w_prep).squeeze(2) + self.bias_prep
        gru_input.relu_()
        ## gru_input is (sample x feature x prep_hidden)
        gru_input    = gru_input.permute(2, 0, 1)
        gru_input    = (gru_input * M_obs).permute(1, 2, 0).contiguous().view(-1, self.prep_hidden * self.input_size)

        temp = h.clone()
        temp[i_obs] = self.gru_d(gru_input, h[i_obs])
        h = temp

        return h

class GRUODECell(nn.Module):
    
    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super(GRUODECell, self).__init__()

        self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t, h):
        """
        Executes one step with autonomous GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            t        time of evaluation
            h        hidden state (current)

        Returns:
            Updated h
        """
        #xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        return dh

class GRUODEBayes(nn.Module):
    def __init__(self, model_params, **options):
        super().__init__()
        # Extract the settings of the model based on the predefined parameters
        self.input_size = model_params.get('input_size', 2)
        self.hidden_size = model_params.get('hidden_size', 50)
        self.p_hidden = model_params.get('p_hidden', 25)
        self.prep_hidden = model_params.get('prep_hidden', 25)
        self.bias = model_params.get('bias', True)
        self.cov_size = model_params.get('cov_size', 1)
        self.cov_hidden = model_params.get('cov_hidden', 1)
        self.classification_hidden = model_params.get('classification_hidden', 1)
        self.dist_type = model_params.get('dist_type', 'niw')
        self.mixing = model_params.get('mixing', 0.0001)
        self.obs_noise_std = model_params.get('obs_noise_std', 0.01)
        self.reg_coeff = model_params.get('beta', 0.01)
        self.dropout_rate = model_params.get('dropout_rate', 0)
        self.solver = model_params.get('solver', 'euler')
        self.reweighting = model_params.get('reweighting', False)
        self.rewt = model_params.get('reweight_threshold', 1.96)
        self.pop_mean = model_params.get('pop_mean')
        self.pop_std = model_params.get('pop_std')
        

        self.impute = False

        if self.dist_type == 'log_normal':
            # The log-normal module is in place to preserve the option to develop a 1-D evidential distribution
            self.p_model = log_normal_dist(self.hidden_size, self.p_hidden, self.input_size, bias=self.bias, dropout_rate=self.dropout_rate)
        elif self.dist_type == 'niw':
            # Construct a multivariate evidential distribution
            self.p_model = normal_inverse_wishart(self.hidden_size, self.p_hidden, self.input_size, bias=self.bias, dropout_rate=self.dropout_rate)
        else:
            raise ValueError(f"Unknown Distribution type '{self.dist_type}'.")

        # Define the various modules used to define the GRUODEBayes model
        self.classification_model = clf_model(self.hidden_size, self.classification_hidden, 1)  # Classifcation model (for smoothing the latent space -- currently unused)
        # The GRU-ODE cell used for evolving the hidden state
        self.gru_c   = GRUODECell(self.hidden_size, bias = self.bias)  
        # The standard GRU cell used to update the hidden state when features are observed
        self.gru_obs = GRUObsCell(self.input_size, self.hidden_size, self.prep_hidden, bias=self.bias, dist_type=self.dist_type, reweighting=self.reweighting, rewt=self.rewt, pop_mean=self.pop_mean, pop_std=self.pop_std)  
        # The mapping function to initialize the hidden state (h_0) from the static covariates
        self.covariates_map = covariates_mapping(self.cov_size, self.cov_hidden, self.hidden_size, bias=self.bias, dropout_rate=self.dropout_rate)

        assert self.solver in ["euler", "midpoint", "dopri5"], "Solver must be either 'euler' or 'midpoint' or 'dopri5'."

        self.store_hist = options.pop("store_hist",False)

        self.apply(init_weights)

    def ode_step(self, h, p, delta_t, current_time, dist_type='log_normal'):
        """Executes a single ODE step."""
        eval_times = torch.tensor([0],device = h.device, dtype = torch.float64)
        eval_ps = torch.tensor([0],device = h.device, dtype = torch.float32)
        if self.impute is False:
            if dist_type == 'log_normal':
                p = torch.zeros_like(p)
            else:
                mu, lmbda, psi, nu = p
                p_mu = torch.zeros_like(mu)
                p_lmbda = torch.zeros_like(lmbda)
                p_psi = torch.zeros_like(psi)
                p_nu = torch.zeros_like(nu)
                p = (p_mu, p_lmbda, p_psi, p_nu)
            
        if self.solver == "euler":
            h = h + delta_t * self.gru_c(p, h)
            p = self.p_model(h)

        elif self.solver == "midpoint":
            k  = h + delta_t / 2 * self.gru_c(p, h)
            pk = self.p_model(k)

            h = h + delta_t * self.gru_c(pk, k)
            p = self.p_model(h)

        elif self.solver == "dopri5":
            assert self.impute==False #Dopri5 solver is only compatible with autonomous ODE.
            solution, eval_times, eval_vals = odeint(self.gru_c,h,torch.tensor([0,delta_t]),method=self.solver,options={"store_hist":self.store_hist})
            if self.store_hist:
                eval_ps = self.p_model(torch.stack([ev[0] for ev in eval_vals]))
            eval_times = torch.stack(eval_times) + current_time
            h = solution[1,:,:]
            p = self.p_model(h)
        else:
            raise ValueError(f"Unknown solver '{self.solver}'.")
        
        current_time += delta_t
        return h,p,current_time, eval_times, eval_ps

        

    def forward(self, times, time_ptr, X, M, obs_idx, delta_t, T, cov, pat_idx, return_path=False):
        """
        Args:
            times      np vector of observation times
            time_ptr   start indices of data for a given time
            X          data tensor
            M          mask tensor (1.0 if observed, 0.0 if unobserved)
            obs_idx    observed patients of each datapoint (indexed within the current minibatch)
            delta_t    time step for Euler
            T          total time
            cov        static covariates for learning the first h0
            pat_idx    the dataset indices for each trajectory (for debugging purposes)
            return_path   whether to return the path of h

        Returns:
            h          hidden state at final time (T)
            loss       loss of the Gaussian observations
        """

        h = self.covariates_map(cov)

        p            = self.p_model(h)
        current_time = 0.0

        loss = 0   # Total loss
        loss_1 = 0 # Pre-jump loss  (Negative Log Likelihood)
        loss_2 = 0 # Post-jump loss (KL between p_updated and the actual sample)
        loss_reg = 0 # Regularization term (Evidential Regularization)

        if return_path:
            path_t = [0]
            path_h = [h]
            if self.dist_type == 'log_normal':
                path_p = [p]
            elif self.dist_type == 'niw':
                mu, lmbda, psi_logvar, nu = p
                path_mu = [mu]
                path_lmbda = [lmbda]
                path_psi = [psi_logvar]
                path_nu = [nu]

        assert len(times) + 1 == len(time_ptr)
        assert (len(times) == 0) or (times[-1] <= T)

        eval_times_total = torch.tensor([],dtype = torch.float64, device = h.device)
        eval_vals_total  = torch.tensor([],dtype = torch.float32, device = h.device)

        for i, obs_time in enumerate(times):
            ## Propagation of the ODE until next observation
            while current_time < (obs_time-0.001*delta_t): #0.0001 delta_t used for numerical consistency.
                 
                if self.solver == "dopri5":
                    h, p, current_time, eval_times, eval_ps = self.ode_step(h, p, obs_time-current_time, current_time, dist_type=self.dist_type)
                else:
                    h, p, current_time, eval_times, eval_ps = self.ode_step(h, p, delta_t, current_time, dist_type=self.dist_type)
                eval_times_total = torch.cat((eval_times_total, eval_times))
                eval_vals_total  = torch.cat((eval_vals_total, eval_ps))

                #Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_h.append(h)
                    if self.dist_type == 'log_normal':
                        path_p.append(p)
                    elif self.dist_type == 'niw':
                        mu, lmbda, psi_logvar, nu = p
                        path_mu.append(mu)
                        path_lmbda.append(lmbda)
                        path_psi.append(psi_logvar)
                        path_nu.append(nu)

            ## Reached an observation
            start = time_ptr[i]
            end   = time_ptr[i+1]

            X_obs = X[start:end]
            M_obs = M[start:end]
            i_obs = obs_idx[start:end]

            # Compute the NLL of the evidential distribution prior to updating the hidden state
            # First, extract the components of the distribution
            if self.dist_type == 'log_normal':
                p_nll = p.clone()
                p_obs        = p_nll[i_obs]
                mean, logvar = torch.chunk(p_obs, 2, dim=1)

                losses = log_normal_nll(X_obs, mean, logvar, M_obs)  ## log normal loss, over all observations

            elif self.dist_type == 'niw':
                p_nll = p
                mean, lmbda, logvar, nu = p_nll
                mean_obs = mean[i_obs]
                lmbda_obs = lmbda[i_obs]
                logvar_obs = logvar[i_obs]
                nu_obs = nu[i_obs]
                
                losses = niw_nll(X_obs, mean_obs, logvar_obs, lmbda_obs, nu_obs, M_obs)

            if losses.sum()!=losses.sum():
                i_pats = np.array(pat_idx)[np.array(i_obs)]
                off_pat_idx = i_pats[torch.where(losses.detach().cpu().isnan())[0]]
                print(f"NaN reached for patient idx {off_pat_idx} at hour {i+1}")
                breakpoint()
                losses_2 = niw_nll(X_obs, mean_obs, logvar_obs, lmbda_obs, nu_obs, M_obs)

            ## Using GRUObservationCell to update h.
            h = self.gru_obs(h, p, X_obs, M_obs, i_obs)

            # Aggregate loss across batches, each iteration will sum across the batch
            loss_1    = loss_1 + losses.sum()
            
            # Update the predictive distribution from the updated hidden state
            p         = self.p_model(h)
            if self.dist_type == 'niw':
                mean, lmbda, logvar, nu = p
                mean_obs = mean[i_obs]
                lmbda_obs = lmbda[i_obs]
                logvar_obs = logvar[i_obs]
                nu_obs = nu[i_obs]
                p_obs = (mean_obs, lmbda_obs, logvar_obs, nu_obs)
                # Compute the Evidential Regularization Term
                if self.reg_coeff > 0:
                    loss_reg = loss_reg + (niw_reg_error(X_obs, mean_obs, lmbda_obs, logvar_obs, nu_obs, mask=M_obs)).sum()
            else:
                p_obs = p[i_obs]

            # Compute the KL loss term
            if self.mixing > 0:
                loss_2 = loss_2 + compute_KL_loss(p_obs = p_obs, X_obs = X_obs, M_obs = M_obs, obs_noise_std=self.obs_noise_std)

            if return_path:
                path_t.append(obs_time)
                path_h.append(h)
                if self.dist_type == 'log_normal':
                    path_p.append(p)
                elif self.dist_type == 'niw':
                    mu, lmbda, psi_logvar, nu = p
                    path_mu.append(mu)
                    path_lmbda.append(lmbda)
                    path_psi.append(psi_logvar)
                    path_nu.append(nu)

        ## after every observation has been processed, propagating until T
        while current_time < T:
            if self.solver == "dopri5":
                h, p, current_time,eval_times, eval_ps = self.ode_step(h, p, T-current_time, current_time, dist_type=self.dist_type)
            else:
                h, p, current_time,eval_times, eval_ps = self.ode_step(h, p, delta_t, current_time, dist_type=self.dist_type)
            eval_times_total = torch.cat((eval_times_total,eval_times))
            eval_vals_total  = torch.cat((eval_vals_total, eval_ps))
            
            #Storing the predictions
            if return_path:
                path_t.append(current_time)
                path_h.append(h)
                if self.dist_type == 'log_normal':
                    path_p.append(p)    
                elif self.dist_type == 'niw':
                    mu, lmbda, psi_logvar, nu = p
                    path_mu.append(mu)
                    path_lmbda.append(lmbda)
                    path_psi.append(psi_logvar)
                    path_nu.append(nu)
                else:
                    pass
        
        loss += loss_1
        
        if self.mixing > 0:  # If we're applying the KL regularization
            loss = loss + (self.mixing * loss_2)
        
        if self.reg_coeff > 0: # If we're applying the Evidential Regularization
            loss = loss + (self.reg_coeff * loss_reg)
       
        if return_path:
            if self.dist_type == 'log_normal':
                return h, loss, np.array(path_t), torch.stack(path_p), torch.stack(path_h), eval_times_total, eval_vals_total
            elif self.dist_type == 'niw':
                return h, loss, np.array(path_t), torch.stack(path_mu), torch.stack(path_lmbda), torch.stack(path_psi), torch.stack(path_nu), torch.stack(path_h), eval_times_total, eval_vals_total
            else:
                pass
        else:
            return h, loss, loss_1, loss_2, loss_reg

############################################
###  Interpolation Prediction Networks
############################################

class IPN(torch.nn.Module):
    def __init__(self, ninp, nhid, nref):
        super(IPN, self).__init__()
        self.nref = nref # number of points to interpolate
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = 1

        # --- Load sub-networks ---
        self.Interpolator = Interpolator(ninp)
        self.rnn = nn.GRU(ninp*3, nhid, batch_first=True)
    
    def forward(self, times, time_ptr, X, M, obs_idx, delta_t, T, cov, pat_idx):
        """
        Args:
            times      np vector of observation times
            time_ptr   start indices of data for a given time
            X          data tensor
            M          mask tensor (1.0 if observed, 0.0 if unobserved)
            obs_idx    observed patients of each datapoint (indexed within the current minibatch)
            delta_t    time step for Euler
            T          total time
            cov        static covariates for learning the first h0
            pat_idx    the dataset indices for each trajectory (for debugging purposes)
            return_path   whether to return the path of h

        Returns:
            h          hidden state at final time (T)
            loss       loss of the Gaussian observations
        """
        obss = M.nonzero()
        obss = torch.concat((obss, obs_idx[obss[:, 0].long()].to(obss.device).unsqueeze(-1)), dim = -1)
        max_n_obs = torch.bincount(obss[:, 2]).max()
        obs_vals = X[obss[:, 0], obss[:, 1]]
        obss = torch.concat((obss, obs_vals.unsqueeze(-1)), dim = -1) # index in X, feature col in X, obs_idx, obs_val
        t_arr = torch.repeat_interleave(torch.from_numpy(times), torch.from_numpy(time_ptr).diff(), 0)
        # index in X, feature col in X, obs_idx, obs_val, time
        obss = torch.concat((obss, t_arr.unsqueeze(-1).to(obss.device)[obss[:, 0].long()]), dim = -1) 

        X = []
        for i in np.arange(0, len(pat_idx)): # slow
            S_i = obss[obss[:, 2] == i][:, (4, 3, 1)] # time, val, dim
            reference_timesteps = torch.linspace(S_i[:, 0].min(), S_i[:, 0].max(), self.nref).unsqueeze(0).to(S_i.device)

            # Interpolation
            x_interpolated = self.Interpolator(S_i, reference_timesteps) # output should be of shape (batch, timesteps, dimensions)
            X.append(x_interpolated.squeeze(0))

            # Prediction
            # state = torch.zeros(self.nlayers, 1, self.nhid)
            # print(x_interpolated.shape)
        X = torch.stack(X)
        out, state = self.rnn(X)
            # logits = self.classification_model(out[:, -1])
        hT = out[:, -1]
        return hT
    
    def computeLoss(self, logits, y):
        # --- save class-specific means ---
        #for i in y.unique():
        #    self.means[i] = self.glimpses[y == i].mean(0).unsqueeze(0)
        return F.cross_entropy(logits, y)

############################################
###  Set Functions for Time Series
############################################

def compute_time_embedding(ts, max_time, num_timescales): # ts is batch_size x num_obs
    timescales = max_time ** torch.linspace(0, 1, num_timescales).to(ts.device)
    scaled_time = ts.unsqueeze(-1) / timescales[None, None, :]
    signal = torch.concat(
            [
                torch.sin(scaled_time),
                torch.cos(scaled_time)
            ],
            dim = -1)
    return signal # batch_size x num_obs x num_timescales*2

class MLP(nn.Module):
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'],hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

class SeFTNetwork(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.num_timescales = hparams['num_timescales']
        self.n_inputs = self.num_timescales*2 + 2
        assert hparams['hidden_size'] % hparams['attn_n_heads'] == 0

        hidden_size_each = int(hparams['hidden_size']/hparams['attn_n_heads'])
        self.encoder = MLP(n_inputs = self.n_inputs, n_outputs =hidden_size_each , hparams = {
            'mlp_width': hparams['encoder_mlp_width'],
            'mlp_dropout': hparams['encoder_mlp_dropout'],
            'mlp_depth': hparams['encoder_mlp_depth']
        })
        
        self.hparams = hparams
        self.max_time = hparams['max_time']
        self.attention = nn.MultiheadAttention(hidden_size_each, num_heads = hparams['attn_n_heads'],
                                                dropout = hparams['attn_dropout'], batch_first = True)
        
    def forward(self, times, time_ptr, X, M, obs_idx, delta_t, T, cov, pat_idx):
        obss = M.nonzero()
        obss = torch.concat((obss, obs_idx[obss[:, 0].long()].to(obss.device).unsqueeze(-1)), dim = -1)
        max_n_obs = torch.bincount(obss[:, 2]).max()
        obs_vals = X[obss[:, 0], obss[:, 1]]
        obss = torch.concat((obss, obs_vals.unsqueeze(-1)), dim = -1) # index in X, feature col in X, obs_idx, obs_val
        t_arr = torch.repeat_interleave(torch.from_numpy(times), torch.from_numpy(time_ptr).diff(), 0)
        # index in X, feature col in X, obs_idx, obs_val, time
        obss = torch.concat((obss, t_arr.unsqueeze(-1).to(obss.device)[obss[:, 0].long()]), dim = -1) 
        S = torch.zeros(len(pat_idx), max_n_obs, 3).to(obss.device) # modality, value, time

        lens = []
        attn_mask = []
        for i in np.arange(0, len(pat_idx)): # slow
            S_i = obss[obss[:, 2] == i][:, (1, 3, 4)]
            lens.append(len(S_i))
            S[i, :lens[-1], :] = S_i
            attn_mask.append([False]*lens[-1] + [True]*(max_n_obs - lens[-1]))
        
        attn_mask = torch.tensor(attn_mask).bool().to(obss.device)
        new_X = torch.concat((compute_time_embedding(S[:, :, -1], self.max_time, self.num_timescales), S[:, :, 0:-1]), dim = -1)        
        encoded_X = self.encoder(new_X)
        _, attn_weights = self.attention(encoded_X, encoded_X, encoded_X, key_padding_mask = attn_mask, average_attn_weights=False)
        attn_weights = attn_weights[:, :, -1, :] # target sequence length 1
        encoded_X_rep = torch.repeat_interleave(encoded_X.unsqueeze(1), self.hparams['attn_n_heads'], dim = 1)
        summed_embeds = (attn_weights.unsqueeze(-1)* encoded_X_rep).sum(dim = 2)
        concat_embeds = summed_embeds.reshape(summed_embeds.shape[0], -1)
        return concat_embeds      


############################################
###  GRU-D
############################################

class GRUD(nn.Module):
    def __init__(self, hparams, input_size):
        super().__init__()
        self.n_layers = hparams['n_layers']
        self.latent_dim = hparams['hidden_size']
        self.grud = GRUD_Layer(input_size, self.latent_dim)
        if self.n_layers > 1:
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, self.n_layers - 1, batch_first = True, 
                dropout = hparams['dropout'], bidirectional = False)
    
    def get_time_delta(self, mask, times):
        B, T, M = mask.shape
        delta_t = torch.concat((torch.tensor([0]), torch.from_numpy(times))).diff().to(mask.device).unsqueeze(0).unsqueeze(-1)
        mask = mask.bool().detach().clone()
        missing_mask = ~mask
        missing_mask = missing_mask.float() * delta_t
        csum1 = torch.cumsum(missing_mask, dim = 1)
        csum2 = csum1.detach().clone()
        csum2[mask] = 0
        cmax = csum2.cummax(dim = 1)[0]
        subtract_mask = -torch.diff(cmax, dim = 1, prepend = torch.zeros(B, 1, M).to(mask.device))
        missing_mask[mask] = subtract_mask[mask]
        csum3 =  torch.cumsum(missing_mask, dim = 1)
        return csum3 + delta_t
    
    def forward(self, times, X, M, cov):
        # X: batch size x n_time x n_features
        delta = self.get_time_delta(M, times)
        h0 = torch.concat((cov, torch.zeros(cov.shape[0], self.latent_dim - cov.shape[1]).to(cov.device)), dim = -1)
        hiddens = self.grud((X, M.float(), delta), h0 = h0)
        if self.n_layers > 1:
            hiddens, _ = self.gru(hiddens)
        return hiddens[-1:, :, :].squeeze(0)


############################################
###       MODEL DEFINITION WRAPPER
############################################

def define_model(params, device):

    model_type = params.get('model_type', 'gruode')

    if model_type == 'gruode':
        model = GRUODEBayes(params)
  
    model.to(device)

    return model


