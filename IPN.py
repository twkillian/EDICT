import torch

class Interpolator(torch.nn.Module):
    def __init__(self, ninp, a=50, k=10.):
        super(Interpolator, self).__init__()
        self.ninp = ninp
        self.k = k # Controls how much alpha gets scaled in coarse interpolation
        
        # --- Parameters ---
        self.alpha = [a]
        self.alpha = torch.nn.Parameter(torch.tensor(self.alpha, dtype=torch.float), requires_grad=True)
        self.rho = torch.nn.Parameter(torch.ones((ninp, ninp)))
        
    def squaredExponentialKernel(self, r, t, alpha):
        dist = torch.exp(torch.mul(-alpha, 1*torch.sub(r, t).pow(2)))
        mask = torch.zeros_like(t)
        mask[t > 0] = 1 #
        return dist*mask + 1e-07 # overflow

    def intensity(self, reference_timesteps, timesteps, alpha, reduce=True):
        if len(reference_timesteps.shape) == 2: # Already batched
            reference_broadcast = torch.repeat_interleave(reference_timesteps.unsqueeze(2), timesteps.shape[1], dim=2)
        elif len(reference_timesteps.shape) == 1: # Just one vector
            reference_broadcast = reference_timesteps.view(-1, 1).repeat(1, timesteps.shape[1]).repeat(timesteps.shape[0], 1, 1)
        else:
            print("wrong reference timestep shape")
        
        timesteps = timesteps.unsqueeze(1) # Add R dim to real timesteps
        # print(timesteps)
        # print(timesteps.shape)
        reference_broadcast = reference_broadcast.unsqueeze(3).repeat(1, 1, 1, timesteps.shape[-1])
        # print(reference_broadcast[:, :10])
        # print(reference_broadcast.unique())
        dist = self.squaredExponentialKernel(reference_broadcast, timesteps, alpha)
        dist = dist/dist.shape[1]
        if reduce:
            return dist.sum(2)
        else:
            return dist

    def interpolate(self, reference_timesteps, timesteps, values, smooth):
        """Compute new values for each reference timestep"""
        if smooth:
            a = self.alpha
        else:
            a = self.k*self.alpha
        lam = self.intensity(reference_timesteps, timesteps, a, reduce=True)
        weights = self.intensity(reference_timesteps, timesteps, a, reduce=False)
        return torch.sum(weights * values.unsqueeze(1), 2)/lam
    
    def crossDimensionInterpolation(self, lam, sigma):
        return torch.matmul(lam*sigma, self.rho)/lam.sum(2, keepdim=True)

    def forward(self, S, reference_timesteps):
        """Params are updated as the model learns"""
        timesteps = S[:, 0].unsqueeze(0)
        values = S[:, 1].unsqueeze(0)
        dimensions = S[:, 2].unsqueeze(0)
        smooth = torch.zeros((1, reference_timesteps.shape[1], self.ninp)).to(reference_timesteps.device)
        coarse = torch.zeros((1, reference_timesteps.shape[1], self.ninp)).to(reference_timesteps.device)
        lam = torch.zeros((1, reference_timesteps.shape[1], self.ninp)).to(reference_timesteps.device)
        for d in range(self.ninp):
            t_d = timesteps[dimensions == d].reshape(1, -1, 1)
            v_d = values[dimensions == d].reshape(1, -1, 1)
            # if t_d[:, 0, :] != 0:
            t_d = torch.concat((torch.zeros((1, 1, 1)).to(t_d.device), t_d), dim=1)
            v_d = torch.concat((torch.zeros((1, 1, 1)).to(v_d.device), v_d), dim=1)
            if len(t_d) > 0:
                l = self.intensity(reference_timesteps, t_d, self.alpha, reduce=True).transpose(2, 1)
                s = self.interpolate(reference_timesteps, t_d, v_d, smooth=True).transpose(2, 1)
                c = self.interpolate(reference_timesteps, t_d, v_d, smooth=False).transpose(2, 1)
                lam[:, :, d] = l
                smooth[:, :, d] = s
                coarse[:, :, d] = c
        cross_dim = self.crossDimensionInterpolation(lam, smooth)
        transient = coarse - cross_dim
        out =  torch.concat([lam, cross_dim, transient], 2)
        return out
