import torch
from torch import nn, optim
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
from sklearn import datasets, preprocessing
from scipy.optimize import minimize_scalar

probit = torch.distributions.normal.Normal(0., 1.).cdf


def g_fun(x):
    return (np.exp(x) - 1. - x)/(x**2)


#### Supervised Policy/Logging Policy training ########

class SupervisedPolicy(pl.LightningModule):
    def __init__(self, n_actions, context_dim, reg, softmax = False, multilabel = False, device = torch.device("cpu")):
        super().__init__()
        self.linear = nn.Linear(context_dim, n_actions, bias=False).to(device)
        self.dev = device
        self.loss_fun = torch.nn.CrossEntropyLoss()
        self.reg = reg
        self.a = n_actions
        self.mask = torch.eye(self.a, dtype=bool).view(1, self.a, self.a, 1)
        self.softmax = softmax
        self.alpha = 1.
        self.multilabel = multilabel
        self.logsoftmax = nn.LogSoftmax(dim = -1)
        
    def policy_a(self, x, a, n_samples = 32):
        
        bs = x.size(0)
        
        scores = self.alpha * self.linear(x)
        
        if self.softmax :
            probs = torch.softmax(scores, dim = 1)
            return probs[torch.arange(bs), a]
        
        scores_a = scores[torch.arange(bs), a].unsqueeze(-1)
        
        diff = scores_a - scores
        
        indices = torch.ones_like(diff).scatter_(1, a.unsqueeze(1), 0.).bool()
        diffs_masked = diff[indices].reshape(bs, self.a - 1, 1)
        
        eps = torch.randn(bs, 1, n_samples)
        diffs_stoch = eps + diffs_masked 
        
        dist_x_a = torch.mean(torch.prod(probit(diffs_stoch), dim = -2), dim = -1)
        
        return dist_x_a
    
    
    def policy(self, x, n_samples = 32):

        bs = x.size(0)
        scores = self.alpha * self.linear(x)
        
        if self.softmax :
            probs = torch.softmax(scores, dim = 1)
            return probs
        
        eps = torch.randn(bs, 1, 1, n_samples)
        diffs = (scores.unsqueeze(-1) - scores.unsqueeze(1)).unsqueeze(-1)
        diffs_masked = diffs.masked_select(~self.mask).view(bs, self.a, self.a - 1, 1)
        
        diffs_stoch = eps + diffs_masked 
        
        dist_x = torch.mean(torch.prod(probit(diffs_stoch), dim = -2), dim = -1)
        
        return dist_x
    
    def forward(self, x):
        dist_x = self.policy(x, n_samples = 1024)
        return dist_x
    

    def sample(self, x):
        scores = self.alpha * self.linear(x)
        eps = torch.randn_like(scores) if not self.softmax else -torch.log(-torch.log(torch.rand_like(scores)))
        return torch.argmax(scores + eps, dim = 1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=.1, weight_decay=self.reg)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        if self.multilabel : 
            logsoftmax = self.logsoftmax(self.linear(x))
            loss = - torch.mean(y * logsoftmax)
        else : 
            loss = self.loss_fun(self.linear(x), y)
        return loss    
    
    

################## Gaussian PBP #######################

class OfflineBanditProbitPolicy(pl.LightningModule):
    
    def __init__(self, n_actions, context_dim, tau, N, lmbd, diag = False, loc_weight=None, device = torch.device("cpu")):
        super().__init__()
        
        if loc_weight is not None: 
            cloned_loc = torch.clone(loc_weight)
            self.q_mean = nn.Parameter(data = cloned_loc)
        else : 
            self.q_mean = nn.Parameter(data = 0.01 * torch.randn(context_dim, n_actions))
        
        if diag : 
            self.q_log_sigma = nn.Parameter(data = torch.zeros_like(self.q_mean))
        else :
            self.log_scale = nn.Parameter(data = torch.zeros(()))
        
        self.diag = diag
        self.tau = tau
        self.d = context_dim
        self.a = n_actions
        self.lmbd = lmbd
        self.mu_0 = torch.clone(cloned_loc).to(device)
        self.N = N
        self.dev = device
        
    def policy_a(self, x, a, n_samples = 32):
        
        bs = x.size(0)
        helper = torch.arange(bs)
        
        if self.diag:
            normalizer = torch.matmul(x**2, torch.exp(2. * self.q_log_sigma).T) ** .5
        else :
            normalizer = torch.ones([bs, self.a]).to(self.dev) * torch.exp(self.log_scale)
        
        normalizer_a = normalizer[helper, a].unsqueeze(-1)
        
        scores = torch.matmul(x, self.q_mean.T)
        scores_a = scores[helper, a].unsqueeze(-1)
        
        diff = (scores_a - scores).unsqueeze(-1)
        sigma_eps = torch.randn(bs, 1, n_samples).to(self.dev) * normalizer_a.view(bs, 1, 1)
        
        diffs_stoch = (sigma_eps + diff)/(normalizer.unsqueeze(-1))
        
        indices = torch.ones_like(diffs_stoch, dtype = bool)
        indices[helper, a] = False
        
        diffs_masked = diffs_stoch[indices].reshape(bs, self.a - 1, n_samples)
        
        dist_x_a = torch.mean(torch.prod(probit(diffs_masked), dim = -2), dim = -1)
        
        return dist_x_a

    def policy(self, x, n_samples = 32):
        
        bs = x.size(0)
        if self.diag:
            normalizer = torch.matmul(x**2, torch.exp(2. * self.q_log_sigma).T) ** .5
        else :
            normalizer = torch.ones([bs, self.a]).to(self.dev) * torch.exp(self.log_scale)
        
        scores = torch.matmul(x, self.q_mean.T)
        
        sigma_eps = torch.randn(bs, 1, 1, n_samples).to(self.dev) * normalizer.view(bs, self.a, 1, 1)
        
        diffs = (scores.unsqueeze(-1) - scores.unsqueeze(1)).unsqueeze(-1)

        diffs_stoch = (sigma_eps + diffs)/normalizer.view(bs, self.a, 1, 1)
        
        prob_diffs = probit(diffs_stoch)
        prob_diffs.diagonal(dim1=1, dim2=2).fill_(1.)
        dist_x = torch.mean(torch.prod(prob_diffs, dim = -2), dim = -1)
        
        return dist_x

    def normal_kl_div(self):
        if self.diag :
            v_part = torch.sum(torch.exp(2. * self.q_log_sigma) - 2. * self.q_log_sigma - 1.)
        else : 
            v_part = self.a * self.d * (torch.exp(2. * self.log_scale) - 2. * self.log_scale - 1.)
            
        m_part = torch.sum((self.q_mean - self.mu_0) ** 2)
        kl_div = 0.5 * (v_part + m_part)
        return kl_div
    
    def compute_mean_second_moment(self, dist_x, ps):
        clipped_ps = torch.where(ps < self.tau, self.tau * torch.ones_like(ps), ps)
        
        sc_moment = (ps * dist_x)/(clipped_ps ** 2)
        second_moment = torch.sum(sc_moment, dim = 1)
        
        return torch.mean(second_moment)

    def compute_bias_term(self, dist_x, ps):
        
        bias_vector = torch.where(ps < self.tau, 1 - ps/self.tau, torch.zeros_like(ps))
        return torch.mean(torch.sum(dist_x * bias_vector, dim = 1))
         

    def forward(self, x):
        dist_x = self.policy(x, n_samples = 512)
        return dist_x


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def compute_risk(self, dist_x_a, a, ps, r):
        
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]
        clipped_p_a = torch.where(p_a < self.tau, self.tau * torch.ones_like(p_a), p_a)
        
        w = dist_x_a/clipped_p_a
        
        risk = r * w 
        
        return risk
        
    def training_step(self, train_batch, batch_idx):
        
        x, a, p_a, r = train_batch
        bsize = x.size(0)
        
        x = x.view(bsize, -1)
        dist_x_a = self.policy_a(x, a)
        
        risk_batch = self.compute_risk(dist_x_a, a, p_a, r)
        loss = torch.mean(risk_batch) + self.lmbd * self.normal_kl_div()
        return loss
    

#####################################################################################################
#####################################################################################################
################################# BERNSTEIN BOUND ######################################################

class OfflineBanditProbitPolicyBernsteinTrained(OfflineBanditProbitPolicy):
    
    def __init__(self, n_actions, context_dim, tau, N, diag = False, loc_weight=None, delta = 0.05, num_p = 200, xi = -0.5, rc = 1., 
                 device = torch.device("cpu")):
        
        super().__init__(n_actions = n_actions, context_dim = context_dim, 
                         tau = tau, N = N, lmbd = 1., diag = diag, loc_weight = loc_weight, device = device)
        
        
        self.delta = delta
        self.num_p = num_p
        self.rc = rc
        
        self.unc_kl1 = np.log((4. * self.N ** 0.5)/self.delta)
        self.unc_kl2 = np.log((2. * self.num_p)/self.delta)
        
        self.xi = xi
        self.tau_xi = (1. + xi)/self.tau - xi
        self.xi_coef = np.maximum(self.xi ** 2, (1 + self.xi) ** 2)
        
        lb = (2. * self.tau * np.log(1/self.delta)/(5. * self.rc * self.N * self.xi_coef)) ** 0.5
        ub = 2. * self.rc / self.tau_xi
       
        self.lmbd_cands = np.linspace(lb, ub, num = num_p)
        
        
    def compute_risk(self, dist_x_a, a, ps, r):
        
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]
        clipped_p_a = torch.where(p_a < self.tau, self.tau * torch.ones_like(p_a), p_a)
        
        w = dist_x_a/clipped_p_a
        
        risk = (r - self.xi) * w + self.xi
        
        return risk
    
        
    
    def find_lmbd_from_candidates(self, kl_a, mm):
        
        g_values = g_fun(self.lmbd_cands * self.tau_xi)
        
        values_1 = kl_a/self.lmbd_cands
        values_2 = self.xi_coef * self.lmbd_cands * g_values * mm
        
        to_min = values_1 + values_2
        lmbd_index = np.argmin(to_min)
        
        return self.lmbd_cands[lmbd_index]
    
    
    def bernstein_batch(self, x, a, ps, r, n_samples = 32):
        
        bsize = x.size(0)
        dist_x = self.policy(x, n_samples = n_samples)
        dist_x_a = dist_x[np.arange(bsize), a]
        
        mm = self.compute_mean_second_moment(dist_x, ps)
        bias = self.compute_bias_term(dist_x, ps)
        risk_mean = torch.mean(self.compute_risk(dist_x_a, a, ps, r))
        
        kl_c = ((self.normal_kl_div() + self.unc_kl1)/(2. * self.N)) ** 0.5
        kl_a = (self.normal_kl_div() + self.unc_kl2)/(self.rc * self.N)
        
        lmbd = self.find_lmbd_from_candidates(kl_a.detach().item(), mm.detach().item())
        
        first_part = risk_mean - self.xi * bias + kl_c
        second_part = kl_a/lmbd + self.xi_coef * lmbd * g_fun(lmbd * self.tau_xi / self.rc) * mm / self.rc
        
        bernstein_bound = first_part + second_part
        
        return bernstein_bound
    
    def bernstein_all_dataloader(self, dataloader, n_samples = 4):
            
            bandit_size = self.rc * self.N
            kl_c = ((self.normal_kl_div() + self.unc_kl1)/(2. * self.N)) ** 0.5
            kl_a = (self.normal_kl_div() + self.unc_kl2)/(bandit_size)
            
            risk, bias, mm = 0., 0., 0.

            for (x, a, ps, r) in tqdm(dataloader) :

                x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
                bsize = x.size(0)
                dist_x = self.policy(x, n_samples = n_samples)
                dist_x_a = dist_x[np.arange(bsize), a]

                mm += self.compute_mean_second_moment(dist_x, ps) * bsize
                bias += self.compute_bias_term(dist_x, ps) * bsize
                risk += torch.sum(self.compute_risk(dist_x_a, a, ps, r))


            risk_mean, bias_mean, mm_mean = risk/bandit_size, bias/bandit_size, mm/bandit_size
            print('Risk :', risk_mean)
            print('Bias :', bias_mean)
            print('Second Moment :', mm_mean)

            lmbd = self.find_lmbd_from_candidates(kl_a.item(), mm_mean.item())

            first_part = risk_mean - self.xi * bias_mean + kl_c
            second_part = kl_a/lmbd + self.xi_coef * lmbd * g_fun(lmbd * self.tau_xi / self.rc) * mm_mean / self.rc

            bernstein_bound = first_part + second_part

            return bernstein_bound
        
        
    def training_step(self, train_batch, batch_idx):
        
        x, a, ps, r = train_batch
        bern_bound = self.bernstein_batch(x, a, ps, r)
        
        return bern_bound 

#####################################################################################################
#####################################################################################################
################################# Exponential Smoothing BOUND #######################################

class OfflineBanditProbitPolicyAlphaTrained(OfflineBanditProbitPolicy):
    
    def __init__(self, n_actions, context_dim, tau, N, diag = False, loc_weight=None, delta = 0.05, num_p = 200, rc = 1., 
                 device = torch.device("cpu")):
        
        super().__init__(n_actions = n_actions, context_dim = context_dim, 
                         tau = tau, N = N, lmbd = 1., diag = diag, loc_weight = loc_weight, device = device)
        
        
        self.delta = delta
        self.num_p = num_p
        self.rc = rc
        self.alpha = 1. - tau
        
        self.unc_kl1 = np.log((4. * self.N ** 0.5)/self.delta)
        self.unc_kl2 = np.log((2. * self.num_p)/self.delta)

        
        lb = 1/self.N
        ub = 1.

        self.lmbd_cands = np.linspace(lb, ub, num = num_p)
        
        
    def compute_risk(self, dist_x_a, a, ps, r):
        
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]
        smoothed_p_a = p_a ** self.alpha
        
        w = dist_x_a/smoothed_p_a
        
        risk = r * w 
        
        return risk
    

    def compute_alpha_mean_second_moment(self, dist_x, a, ps, r):
        bsize = dist_x.size(0)
        smoothed_ps = ps ** self.alpha
        
        sc_moment = (ps * dist_x)/(smoothed_ps ** 2)
        second_moment = torch.sum(sc_moment, dim = 1)

        p_a = ps[torch.arange(bsize), a]
        dist_x_a = dist_x[torch.arange(bsize), a]
        smoothed_p_a = p_a ** self.alpha
        
        risk_2 = dist_x_a * (r/smoothed_p_a) ** 2
        
        return torch.mean(second_moment + risk_2)
        
    
    def find_lmbd_from_candidates(self, kl_a, mm):
        
        l_values = self.lmbd_cands/2
        
        values_1 = kl_a/self.lmbd_cands
        values_2 = l_values * mm
        
        to_min = values_1 + values_2
        lmbd_index = np.argmin(to_min)
        
        return self.lmbd_cands[lmbd_index]
    
    
    def alpha_bernstein_batch(self, x, a, ps, r, n_samples = 32):
        
        bsize = x.size(0)
        dist_x = self.policy(x, n_samples = n_samples)
        dist_x_a = dist_x[np.arange(bsize), a]
        
        mm = self.compute_alpha_mean_second_moment(dist_x, a, ps, r)
        risk_mean = torch.mean(self.compute_risk(dist_x_a, a, ps, r))
        
        kl_c = ((self.normal_kl_div() + self.unc_kl1)/(2. * self.N)) ** 0.5
        kl_a = (self.normal_kl_div() + self.unc_kl2)/(self.rc * self.N)
        
        lmbd = self.find_lmbd_from_candidates(kl_a.detach().item(), mm.detach().item())
        
        first_part = risk_mean + kl_c
        second_part = kl_a/lmbd + lmbd * mm / 2.
        
        bernstein_bound = first_part + second_part
        
        return bernstein_bound
    
    def alpha_bernstein_all_dataloader(self, dataloader, n_samples = 4):
            
            bandit_size = self.rc * self.N
            kl_c = ((self.normal_kl_div() + self.unc_kl1)/(2. * self.N)) ** 0.5
            kl_a = (self.normal_kl_div() + self.unc_kl2)/(bandit_size)
            
            risk, mm = 0., 0.

            for (x, a, ps, r) in tqdm(dataloader) :

                x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
                bsize = x.size(0)
                dist_x = self.policy(x, n_samples = n_samples)
                dist_x_a = dist_x[np.arange(bsize), a]

                mm += self.compute_alpha_mean_second_moment(dist_x, a, ps, r) * bsize
                risk += torch.sum(self.compute_risk(dist_x_a, a, ps, r))


            risk_mean, mm_mean = risk/bandit_size, mm/bandit_size
            print('Risk :', risk_mean)
            print('Second Moment :', mm_mean)

            lmbd = self.find_lmbd_from_candidates(kl_a.item(), mm_mean.item())

            first_part = risk_mean + kl_c
            second_part = kl_a/lmbd + lmbd * mm_mean / 2.

            bernstein_bound = first_part + second_part

            return bernstein_bound
        
        
    def training_step(self, train_batch, batch_idx):
        
        x, a, ps, r = train_batch
        bern_bound = self.alpha_bernstein_batch(x, a, ps, r)
        
        return bern_bound 
    

#####################################################################################################
#####################################################################################################
################################# IX BOUND ############################################
    

def psi_fun(x, lambd):
    return (1 - torch.exp(-lambd * x))/lambd



class OfflineBanditProbitPolicyIXTrained(OfflineBanditProbitPolicy):
    
    def __init__(self, n_actions, context_dim, tau, N, diag = False, loc_weight=None, delta = 0.05, num_p = 200,  
                 device = torch.device("cpu")):
        
        super().__init__(n_actions = n_actions, context_dim = context_dim, 
                         tau = tau, N = N, lmbd = 1., diag = diag, loc_weight = loc_weight, device = device)
        
        
        self.delta = delta
        self.num_p = num_p
        self.lmbd_cands = np.linspace(1./np.sqrt(N), 1., num = num_p)
        
        self.unc_kl = np.log(num_p/self.delta)
        
        
    def compute_second_moment_ips(self, dist_x_a, a, ps, r):
        
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]
        
        sm_ips = dist_x_a * (r / p_a)**2
        
        return sm_ips
    
        
    def compute_risk(self, dist_x_a, a, ps, r, lambd):
        
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]

        lin_clipping = r/(p_a + lambd/2.)
        
        risk = dist_x_a * lin_clipping
        
        return risk
        
    
    def find_lmbd_from_candidates(self, kl_n, second_moment):
        
        lambd_star = np.sqrt(2 * kl_n.cpu().item() / (second_moment.cpu().item() + 1e-2))
        
        mask = 1. * (self.lmbd_cands > lambd_star)
        try:
          first_index = np.where(mask == 1.)[0][0]
        except:
            first_index = self.lmbd_cands[-1]
        
        return self.lmbd_cands[first_index]
    
    
    def ix_batch(self, x, a, ps, r, n_samples = 32):
        
        kl_n = (self.normal_kl_div() + self.unc_kl)/self.N

        dist_x_a = self.policy_a(x, a, n_samples = n_samples)

        sm_batch = self.compute_second_moment_ips(dist_x_a, a, ps, r)
        sm_mean = torch.mean(sm_batch)
        lambd = self.find_lmbd_from_candidates(kl_n, sm_mean)
        
        risk_mean = torch.mean(self.compute_risk(dist_x_a, a, ps, r, lambd))
        ix_bound = risk_mean + kl_n/lambd
        
        return ix_bound
    
    def ix_all_dataloader(self, dataloader, n_samples = 4):
            
            kl_n = (self.normal_kl_div() + self.unc_kl)/self.N
            
            risk, sm = 0. , 0.

            for (x, a, ps, r) in tqdm(dataloader) :

                x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
                dist_x_a = self.policy_a(x, a, n_samples = n_samples)
                sm += torch.sum(self.compute_second_moment_ips(dist_x_a, a, ps, r))

            sm_mean = sm / self.N
            lambd = self.find_lmbd_from_candidates(kl_n, sm_mean)
            
            for (x, a, ps, r) in tqdm(dataloader) :

                x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
                dist_x_a = self.policy_a(x, a, n_samples = n_samples)
                risk += torch.sum(self.compute_risk(dist_x_a, a, ps, r, lambd))


            risk_mean = risk/self.N
            print('Risk :', risk_mean)

            ix_bound = risk_mean + kl_n/lambd

            return ix_bound
        
        
    def training_step(self, train_batch, batch_idx):
        
        x, a, ps, r = train_batch
        ix_bound = self.ix_batch(x, a, ps, r)
        
        return ix_bound


#####################################################################################################
#####################################################################################################
################################# LS BOUND ######################################################
    

class OfflineBanditProbitPolicyLogTrained(OfflineBanditProbitPolicy):
    
    def __init__(self, n_actions, context_dim, tau, N, diag = False, loc_weight=None, delta = 0.05, num_p = 200,  
                 device = torch.device("cpu")):
        
        super().__init__(n_actions = n_actions, context_dim = context_dim, 
                         tau = tau, N = N, lmbd = 1., diag = diag, loc_weight = loc_weight, device = device)
        
        
        self.delta = delta
        self.num_p = num_p
        self.lmbd_cands = np.linspace(1/np.sqrt(N), 1., num = num_p)
        
        self.unc_kl = np.log(num_p/self.delta)
        self.lambd = 1/np.sqrt(N)
        
        
    def compute_second_moment_ips(self, dist_x_a, a, ps, r):
        
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]
        
        sm_ips = dist_x_a * (r / p_a)**2
        
        return sm_ips
    
        
    def compute_risk(self, dist_x_a, a, ps, r, lambd):
        
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]

        lin_log_smooth = - torch.log(1 - lambd * r / p_a)/lambd
        
        risk = dist_x_a * lin_log_smooth
        
        return risk
        
    
    def find_lmbd_from_candidates(self, kl_n, second_moment):
        
        lambd_star = np.sqrt(2 * kl_n.cpu().item() / (second_moment.cpu().item() + 1e-2))
        
        mask = 1. * (self.lmbd_cands > lambd_star)
        try:
          first_index = np.where(mask == 1.)[0][0]
        except:
            first_index = self.lmbd_cands[-1]
        
        return self.lmbd_cands[first_index]
    
    
    def log_smooth_batch(self, x, a, ps, r, n_samples = 32):
        
        kl_n = (self.normal_kl_div() + self.unc_kl)/self.N

        dist_x_a = self.policy_a(x, a, n_samples = n_samples)

        sm_batch = self.compute_second_moment_ips(dist_x_a, a, ps, r)
        sm_mean = torch.mean(sm_batch)
        lambd = self.find_lmbd_from_candidates(kl_n, sm_mean)
        
        risk_mean = torch.mean(self.compute_risk(dist_x_a, a, ps, r, lambd))
        linear_bound = risk_mean + kl_n/lambd
        log_bound = psi_fun(linear_bound, lambd)
        
        return log_bound
    
    def log_all_dataloader(self, dataloader, n_samples = 4):
            
            kl_n = (self.normal_kl_div() + self.unc_kl)/self.N
            
            risk, sm = 0. , 0.

            for (x, a, ps, r) in tqdm(dataloader) :

                x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
                dist_x_a = self.policy_a(x, a, n_samples = n_samples)
                sm += torch.sum(self.compute_second_moment_ips(dist_x_a, a, ps, r))

            sm_mean = sm / self.N
            lambd = self.find_lmbd_from_candidates(kl_n, sm_mean)
            
            for (x, a, ps, r) in tqdm(dataloader) :

                x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
                dist_x_a = self.policy_a(x, a, n_samples = n_samples)
                risk += torch.sum(self.compute_risk(dist_x_a, a, ps, r, lambd))


            risk_mean = risk/self.N
            print('Risk :', risk_mean)

            linear_bound = risk_mean + kl_n/lambd

            log_bound = psi_fun(linear_bound, lambd)

            return log_bound
        
        
    def training_step(self, train_batch, batch_idx):
        
        x, a, ps, r = train_batch
        log_bound = self.log_smooth_batch(x, a, ps, r)
        
        return log_bound
