import torch
import torch.nn as nn

class GRUD_Layer(torch.nn.Module):
    def __init__(self, ninp, nhid):
        super(GRUD_Layer, self).__init__()
        self._ninp = ninp
        self.nhid = nhid

        self._zeros_x = torch.zeros(self._ninp)
        self._zeros_h = torch.zeros(self.nhid)

        # --- Sub-networks ---
        # self.Classifier = torch.nn.Linear(self.nhid, self._nclasses)
        combined_dim = self.nhid + 2*self._ninp # Input and missingness vector

        # --- mappings ---
        self.z = nn.Linear(combined_dim, self.nhid) # Update gate
        self.r = nn.Linear(combined_dim, self.nhid) # Reset gate
        self.h = nn.Linear(combined_dim, self.nhid) # Hidden (?) gate
        self.gamma_x = torch.nn.Linear(self._ninp, self._ninp)
        self.gamma_h = torch.nn.Linear(self._ninp, self.nhid)

    def unpack(self, data):
        vals, masks, past = data
        vals = vals.transpose(0, 1).float() # Convert to shape T x B x V
        masks = masks.transpose(0, 1).float()
        past = past.transpose(0, 1).float()
        past[torch.isnan(past)] = 0.0
        vals[torch.isnan(vals)] = 0.0
        masks[torch.isnan(masks)] = 1.0
        past[torch.isnan(past)] = 1.0
        return vals, masks, past

    def gru_d_cell(self, x, h, m, dt, x_prime, x_mean):
        # --- compute decays ---
        delta_x = torch.exp(-torch.max(self._zeros_x.to(x.device), self.gamma_x(dt)))

        # --- apply state-decay ---
        delta_h = torch.exp(-torch.max(self._zeros_h.to(x.device), self.gamma_h(dt)))
        h = delta_h * h

        x_prime = m*x + (1-m)*x_prime # Update last-observed value

        # --- estimate new x value ---
        x = m*x + (1-m)*(delta_x*x_prime + (1-delta_x)*x_mean)

        # --- gating functions ---
        combined = torch.cat((x, h, m), dim=1)
        r = torch.sigmoid(self.r(combined))
        z = torch.sigmoid(self.z(combined))
        new_combined = torch.cat((x, torch.mul(r, h), m), dim=1)
        h_tilde = torch.tanh(self.h(new_combined))
        h = (1 - z)*h + z*h_tilde
        return h, x_prime

    def forward(self, data, h0 = None):
        vals, masks, past = self.unpack(data)
        T, B, V = vals.shape

        all_h = []
        if h0 is not None:
            h = h0.detach().clone()
        else:
            h = torch.zeros(B, self.nhid).to(vals.device)
        x_prime = torch.zeros(self._ninp).to(vals.device)
        for t in range(vals.shape[0]):
            x = vals[t]
            m = masks[t]
            d = past[t]
            x_mean = (masks[:(t+1)]*vals[:(t+1)]).sum(0)/masks[:(t+1)].sum(0)
            x_mean[torch.isnan(x_mean)] = 0.0
            h, x_prime = self.gru_d_cell(x, h, m, d, x_prime, x_mean)
            all_h.append(h)

        # logits = self.Classifier(h)
        return torch.stack(all_h)