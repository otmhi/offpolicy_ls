import numpy as np
import torch
from torchvision.datasets import FashionMNIST, MNIST, EMNIST
from torchvision import transforms
from pytorch_lightning import Trainer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from utils import test_risk_exact_probit, build_bandit_dataset
from learning_bounds import SupervisedPolicy, OfflineBanditProbitPolicyBernsteinTrained, OfflineBanditProbitPolicyAlphaTrained, OfflineBanditProbitPolicyIXTrained, OfflineBanditProbitPolicyLogTrained
import pandas as pd
from sklearn.model_selection import train_test_split

import os
import warnings
warnings.filterwarnings("ignore")


# Reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('The device available :', device)

device_pl = "gpu" if torch.cuda.is_available() else "cpu"

dict_datasets = {'MNIST' : MNIST,
                 'FashionMNIST': FashionMNIST,
                 'EMNIST': EMNIST}

# Setting
name = 'MNIST'
number_of_rep = 10

print('dataset used is :', name)


dataset = dict_datasets[name]

if name == 'EMNIST':
    mnist_train = dataset(os.getcwd(), train=True, split = 'balanced', download=True, transform=transforms.ToTensor())
    mnist_test = dataset(os.getcwd(), train=False, split = 'balanced', download=True, transform=transforms.ToTensor())
else :
    mnist_train = dataset(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    mnist_test = dataset(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

X_train, Y_train = torch.tensor(mnist_train.data).float().reshape(len(mnist_train.data), -1), torch.tensor(mnist_train.targets)
X_test, Y_test = torch.tensor(mnist_test.data).float().reshape(len(mnist_test.data), -1), torch.tensor(mnist_test.targets)

# ## Rescaling the contexts
X_train /= torch.norm(X_train, p = 2, dim = -1, keepdim=True)
X_test /= torch.norm(X_test, p = 2, dim = -1, keepdim=True)

# Dataset subsampling ==> Learn a logging Policy

x0, X_log, y0, Y_log = train_test_split(X_train, Y_train, train_size = 0.05)

N = len(X_log)
N_test = len(X_test)
print('Train : dimension of X is :', X_log.shape, 'dimension of Y is :', N)
print('Test : dimension of X_test is :', X_test.shape, 'dimension of Y is :', N_test)
context_dim = X_log.shape[1]
num_actions = len(np.unique(Y_log))
print('num_actions: ', num_actions)

subsample_pt = TensorDataset(x0, y0)
subsample_dataloader = DataLoader(subsample_pt, batch_size=32)

# Create the logging split
logging_split = TensorDataset(X_log, Y_log)
logging_split_dataloader = DataLoader(logging_split, batch_size=32)


# Training a logging policy

alphas = [0.1, 0.3, 0.5, 0.7, 1.]

bounds = ['cIPS - Bernstein', 'ES',  'cvcIPS - Bernstein', 'IX', 'LS']
bounds_rc = ['seed', 'alpha', 'Logging Risk'] + [b for b in bounds]

bern, alpha_smooth, bern_cv, ix, log_s = bounds
dict_results = {key : [] for key in bounds_rc}
epochs_logging = 10
epochs = 10

logging_policy = SupervisedPolicy(n_actions=num_actions, context_dim=context_dim, softmax = True, reg=1e-4, device = device)
trainer = Trainer(max_epochs=epochs_logging, devices=1, accelerator=device_pl, enable_model_summary=False, logger=None, enable_checkpointing=False)
trainer.fit(logging_policy, subsample_dataloader)
seeds = list(np.arange(number_of_rep))

for i, alpha in enumerate(alphas) :
    logging_policy = logging_policy.to(device)
    print('alpha parameter : ', alpha)  
    logging_policy.alpha = alpha
    risk_logging = test_risk_exact_probit(X_test, Y_test, logging_policy)
    print('The exact risk after training - Softmax: ', risk_logging)

    for seed in seeds:
        dict_results['seed'].append(seed)
        dict_results['alpha'].append(alpha)
        dict_results['Logging Risk'].append(risk_logging)
        
        # Collect a bandit dataset
        f, a, p, c = build_bandit_dataset(logging_split_dataloader, logging_policy, replay_count = 1)

        print('max', p.max(dim = 0)[0].mean().item())
        print('min', p.min(dim = 0)[0].mean().item())

        bandit_train_posterior = TensorDataset(f, a, p, c)
        bandit_train_posterior_dataloader = DataLoader(bandit_train_posterior, batch_size=128)

        mu_0 = alpha * logging_policy.linear.weight.data
        tau = 1./num_actions
        

        ######## Bernstein cIPS ##########
        gpbp_bernstein0 = OfflineBanditProbitPolicyBernsteinTrained(n_actions=num_actions, context_dim=context_dim, tau = tau, 
                                                                   N = N, loc_weight = mu_0, num_p = 100, device = device, rc = 1, xi = 0)


        trainer = Trainer(max_epochs=epochs, devices=1, accelerator=device_pl, enable_model_summary=False, logger=False, enable_checkpointing=False)
        trainer.fit(gpbp_bernstein0, bandit_train_posterior_dataloader)


        gpbp_bernstein0 = gpbp_bernstein0.to(device)
        with torch.no_grad():
            guarantees_bernstein0 = gpbp_bernstein0.bernstein_all_dataloader(bandit_train_posterior_dataloader, 1000).item()
            risk_after_train0 = test_risk_exact_probit(X_test, Y_test, gpbp_bernstein0)
        print('--------------------------------------------------------------------------------')
        print('Risk Guarantees after training - Bernstein 0:', guarantees_bernstein0)
        print('True Risk after training  :', risk_after_train0)
        print('--------------------------------------------------------------------------------')
        
        dict_results[bern].append((guarantees_bernstein0, risk_after_train0))
        
        ######################

        ######## ES Bound ##########
        gpbp_alpha = OfflineBanditProbitPolicyAlphaTrained(n_actions=num_actions, context_dim=context_dim, tau = tau, 
                                                           N = N, loc_weight = mu_0, num_p = 100, device = device, rc = 1)


        trainer = Trainer(max_epochs=epochs, devices=1, accelerator=device_pl, enable_model_summary=False, logger=False, enable_checkpointing=False)
        trainer.fit(gpbp_alpha, bandit_train_posterior_dataloader)


        gpbp_alpha = gpbp_alpha.to(device)
        with torch.no_grad():
            guarantees_alpha = gpbp_alpha.alpha_bernstein_all_dataloader(bandit_train_posterior_dataloader, 1000).item()
            risk_after_train0 = test_risk_exact_probit(X_test, Y_test, gpbp_alpha)
        print('--------------------------------------------------------------------------------')
        print('Risk Guarantees after training - Alpha:', guarantees_alpha)
        print('True Risk after training  :', risk_after_train0)
        print('--------------------------------------------------------------------------------')
        
        dict_results[alpha_smooth].append((guarantees_alpha, risk_after_train0))
        
        ######################


        ######## Bernstein cvcIPS ##########
        gpbp_bernstein = OfflineBanditProbitPolicyBernsteinTrained(n_actions=num_actions, context_dim=context_dim, tau = tau, 
                                                                   N = N, loc_weight = mu_0, num_p = 100, device = device, rc = 1)


        trainer = Trainer(max_epochs=epochs, devices=1, accelerator=device_pl, enable_model_summary=False, logger=False, enable_checkpointing=False)
        trainer.fit(gpbp_bernstein, bandit_train_posterior_dataloader)


        gpbp_bernstein = gpbp_bernstein.to(device)
        with torch.no_grad():
            guarantees_bernstein = gpbp_bernstein.bernstein_all_dataloader(bandit_train_posterior_dataloader, 1000).item()
            risk_after_train = test_risk_exact_probit(X_test, Y_test, gpbp_bernstein)
        print('--------------------------------------------------------------------------------')
        print('Risk Guarantees after training - Bernstein -1/2 :', guarantees_bernstein)
        print('True Risk after training  :', risk_after_train)
        print('--------------------------------------------------------------------------------')
        
        dict_results[bern_cv].append((guarantees_bernstein, risk_after_train))
        ######################

        ######## IX ##########
        gpbp_ix  = OfflineBanditProbitPolicyIXTrained(n_actions=num_actions, context_dim=context_dim, tau = tau, 
                                                      N = N, loc_weight = mu_0, num_p = 100, device = device)


        trainer = Trainer(max_epochs=epochs, devices=1, accelerator=device_pl, enable_model_summary=False, logger=False, enable_checkpointing=False)
        trainer.fit(gpbp_ix , bandit_train_posterior_dataloader)


        gpbp_ix = gpbp_ix.to(device)
        with torch.no_grad():
            guarantees_ix = gpbp_ix.ix_all_dataloader(bandit_train_posterior_dataloader, 1000).item()
            risk_after_train0 = test_risk_exact_probit(X_test, Y_test, gpbp_ix)
        print('--------------------------------------------------------------------------------')
        print('Risk Guarantees after training - IX:', guarantees_ix)
        print('True Risk after training  :', risk_after_train0)
        print('--------------------------------------------------------------------------------')
        
        dict_results[ix].append((guarantees_ix, risk_after_train0))
        
        ######################

        ######## LS ##########
        gpbp_logsmooth = OfflineBanditProbitPolicyLogTrained(n_actions=num_actions, context_dim=context_dim, tau = tau, 
                                                              N = N, loc_weight = mu_0, num_p = 100, device = device)


        trainer = Trainer(max_epochs=epochs, devices=1, accelerator=device_pl, enable_model_summary=False, logger=False, enable_checkpointing=False)
        trainer.fit(gpbp_logsmooth, bandit_train_posterior_dataloader)


        gpbp_logsmooth = gpbp_logsmooth.to(device)
        with torch.no_grad():
            guarantees_logsmooth = gpbp_logsmooth.log_all_dataloader(bandit_train_posterior_dataloader, 1000).item()
            risk_after_train0 = test_risk_exact_probit(X_test, Y_test, gpbp_logsmooth)
        print('--------------------------------------------------------------------------------')
        print('Risk Guarantees after training - Log Smooth:', guarantees_logsmooth)
        print('True Risk after training  :', risk_after_train0)
        print('--------------------------------------------------------------------------------')
        
        dict_results[log_s].append((guarantees_logsmooth, risk_after_train0))
        
        ######################
        

    print(dict_results)

df = pd.DataFrame(dict_results)
print(df)
df.to_csv('results_' + name +'.csv', index = False)






