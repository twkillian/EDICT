"""
Loss functions for use in defining and evaluating an evidential distribution over
the likelihood of the predictions provided by the trained model.

Notes:
 - 
"""

############################################
#           IMPORTS and DEFINITIONS
############################################
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

############################################
#   NORMAL INVERSE GAMMA (univariate)
############################################
def nig_nll(y, gamma, v, alpha, beta):
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
            - alpha * torch.log(two_blambda) \
            + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + two_blambda) \
            + torch.lgamma(alpha) \
            - torch.lgamma(alpha + 0.5)

    return nll

def nig_reg(y, gamma, v, alpha, beta):
    error = F.l1_loss(y, gamma, reduction="none")
    evi = 2 * v + alpha
    return error * evi

def log_normal_nll(X, mean, logvar, masks):
    """Compute the log normal loss over all observations..."""
    sigma = torch.exp(0.5*logvar)
    error = (X-mean)/sigma
    log_lik_c = np.log(np.sqrt(2*np.pi))
    losses = 0.5*((torch.pow(error, 2) + logvar + 2*log_lik_c) * masks)
    return losses

############################################
#   NORMAL INVERSE WISHART (multivariate)
############################################
def niw_nll(X, mu, logvar, lmbda, nu, masks=None): 
    """Compute the Negative Log Likelihood for the Normal-Inverse Wishart Distribution"""
    if masks is None:
        # We don't want to remove anything from the computation of the NLL!
        masks = torch.ones_like(X)
    dim = mu.shape[-1]
    
    sigma = torch.sqrt(torch.exp(logvar) / (lmbda*(nu - dim - 1))) 
    error = (X - mu) / sigma
    
    error = (error*masks).unsqueeze(-1)
    psi = torch.zeros(error.shape[0], dim, dim, device=logvar.device)
    psi += torch.diag_embed(torch.exp(logvar), dim1=-2, dim2=-1)
    lam_ratio = (1+lmbda) / lmbda
    psi_factor = psi + (1/lam_ratio).unsqueeze(-1) * (error @ error.transpose(-2, -1))
    psi_factor_det = torch.logdet(psi_factor)

    nll = dim * 0.5 * torch.log(np.pi * lam_ratio) \
        + torch.lgamma((nu - dim + 1) * 0.5) \
        - torch.lgamma((nu + 1) * 0.5) \
        - 0.5 * nu * torch.logdet(psi).unsqueeze(-1) \
        + 0.5 * (nu + 1) * psi_factor_det.unsqueeze(-1)

    return nll

class niw_reg(nn.Module):
    def __init__(self, params):
        super(niw_reg, self).__init__()
        """Define the regularization terms for the Normal-Inverse Wishart Evidential Distribution"""
        self.alpha = params.get('alpha', 0.5)
        self.beta = params.get('beta', 0.5)
        self.missing_exp_factor = params.get('missing_exp_factor', 0.85)
        self.present_exp_factor = params.get('present_exp_factor', 1.5)

        # self.temporal_reg = niw_reg_temporal(missing_exp_factor, present_exp_factor)
        self.temporal_reg = MultVariateKLD("sum")
        self.error_reg = niw_reg_error

    def forward(self, y, mu, lmbda, psi, nu, **kwargs):
        # Return the linear combination of the two regularization components
        # temporal = self.temporal_reg(y, mu, lmbda, psi, nu, **kwargs)
        niw_var = torch.diagonal(torch.exp(psi) / (lmbda * (nu - mu.shape[-1] - 1)).unsqueeze(-1), dim1=-2, dim2=-1)
        # niw_var = torch.sqrt(niw_var).squeeze()  # Compute the square root of the component variances...
        temporal = self.temporal_reg(y, mu, 1e-2*torch.ones(y.shape[-1]), niw_var, **kwargs) # MV KL Term...
        error = self.error_reg(y, mu, lmbda, psi, nu, **kwargs)

        # Should be a linear combination of 1xD vectors for each timestep...
        # Subtracting temporal regularization because we want to maximize the similarity...
        return self.alpha * temporal.mean() + self.beta * error.mean(), temporal.mean().detach(), error.mean().detach()

def niw_reg_error(y, mu, lmbda, psi, nu, mask=None, **kwargs):
    if torch.any(y.isnan()): # Zero fill any and all NaNs
        y = torch.nan_to_num(y, 0.0)
    error = F.l1_loss(y, mu, reduction="none")
    evidence = lmbda + nu  # Account for the virtual observations (e.g. the total evidence)
    if mask is not None:  # Remove features according to the missingness mask
        error = mask * error 
    return (error * evidence).sum(-1)  # Scale errors by the total evidence (should be a 1xD vector per example), sum across dimensions.


############################################
#        KL Divergence Terms
############################################

def compute_KL_loss(p_obs, X_obs, M_obs, obs_noise_std=1e-2, logvar=True):
    obs_noise_std = torch.tensor(obs_noise_std)
    if logvar:
        if isinstance(p_obs, tuple):
            mean, lmbda, logvar, nu = p_obs
            v = (torch.exp(logvar) / (lmbda * (nu - mean.shape[-1] - 1)))
            std = torch.sqrt(v)

            KL_func = MultVariateKLD("sum")
        else: 
            mean, logvar = torch.chunk(p_obs, 2, dim=1)
            std = torch.exp(0.5*logvar)

            KL_func = gaussian_KL
    else:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        ## making var non-negative and also non-zero (by adding a small value)
        std       = torch.pow(torch.abs(var) + 1e-5,0.5)

        KL_func = gaussian_KL

    return KL_func(mu1=X_obs, mu2=mean, sigma1=obs_noise_std*torch.ones(X_obs.shape[-1], device=std.device), sigma2=std, mask=M_obs)
    
def gaussian_KL(mu1, mu2, sigma1, sigma2, mask=None):
    kl_return = (torch.log(sigma2) - torch.log(sigma1) + (torch.pow(sigma1,2)+torch.pow((mu1 - mu2),2)) / (2*sigma2**2) - 0.5)
    # Remove components of the KL that are not derived from observed features
    if mask is not None:
        kl_return = kl_return * mask
    
    return kl_return.sum()

# Define kl loss
class MultVariateKLD(torch.nn.Module):
    def __init__(self, reduction):
        super(MultVariateKLD, self).__init__()
        self.reduction = reduction

    def forward(self, mu1, mu2, sigma1, sigma2, mask=None, **kwargs):
        if torch.any(mu1.isnan()): # Zero fill any and all NaNs (in obs tensor "mu1")
            mu1 = torch.nan_to_num(mu1, 0.0)
        mu1, mu2 = mu1.type(dtype=torch.float64), mu2.type(dtype=torch.float64)
        
        sigma1 = sigma1.type(dtype=torch.float64)
        sigma2 = sigma2.type(dtype=torch.float64)

        sigma_diag_1 = torch.diag_embed(sigma1, offset=0, dim1=-2, dim2=-1)
        sigma_diag_2 = torch.diag_embed(sigma2, offset=0, dim1=-2, dim2=-1)

        sigma_diag_2_inv = sigma_diag_2.inverse()

        mu_diff = (mu2-mu1)
        if mask is not None:  # Remove features according to the missingness mask
            mu_diff = mask*mu_diff
        mu_diff = mu_diff.unsqueeze(-1)

        # log(det(sigma2^T)/det(sigma1))
        term_1 = (sigma_diag_2.det() / sigma_diag_1.det()).log()
        
        # trace(inv(sigma2)*sigma1)
        term_2 = torch.diagonal((torch.matmul(sigma_diag_2_inv, sigma_diag_1)), dim1=-2, dim2=-1).sum(-1)

        # (mu2-m1)^T*inv(sigma2)*(mu2-mu1)
        term_3 = torch.matmul(torch.matmul(mu_diff.transpose(-2, -1), sigma_diag_2_inv),
                              mu_diff).squeeze()

        # dimension of embedded space (number of mus and sigmas)
        n = mu1.shape[-1]

        # Calc kl divergence on entire batch 
        kl = 0.5 * (term_1 - n + term_2 + term_3)

        # Calculate mean kl_d loss
        if self.reduction == 'mean':
            kl_agg = torch.mean(kl)
        elif self.reduction == 'sum':
            kl_agg = torch.sum(kl, dim=0)
        else:
            raise NotImplementedError(f'Reduction type not implemented: {self.reduction}')

        return kl_agg

############################################
#   General loss function wrapper
############################################        

class evidential_regression_loss(nn.Module):
    def __init__(self, params):
        super(evidential_regression_loss, self).__init__()
        self.form = params.get('form', 'niw')
        self.reg_coeff = params.get('mixing', 0.0001)
        # Define the NLL and Regularization losses
        if self.form == 'nig':
            self.loss_nll = nig_nll
            self.loss_reg = nig_reg
        elif self.form == 'niw':
            self.loss_nll = niw_nll
            # Here is where we'll pass the alpha and beta
            # coefficients for the regularization components of the loss
            self.loss_reg = niw_reg(params) 
        else:
            raise ValueError(f"{self.form.upper()} losses are not defined")

    def forward(self, y, pred, **kwargs): 
        # kwargs here is 'time_since_obs' which is a tensor 
        # of time since each feature was last observed 

        # For convenience we'll keep these terms in the NIG naming
        # For NIW gamma = mu, v = lambda, alpha = Psi, beta = nu
        gamma, v, alpha, beta = pred  # Parameters of the NIW distribution
        loss_nll = self.loss_nll(y, gamma, v, alpha, beta, **kwargs)
        loss_reg, temp_reg, error_reg = self.loss_reg(y, gamma, v, alpha, beta, **kwargs)
        return loss_nll.mean() + self.reg_coeff * loss_reg, temp_reg.item(), error_reg.item()
