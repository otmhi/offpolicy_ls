import torch
import numpy as np
from tqdm import tqdm
from sklearn import datasets, preprocessing


def free_gpu(model):
    del model
    torch.cuda.empty_cache()
    

def load_data(data='scene'):
    train_filename = '{}/{}_train.svm'.format(data, data)
    test_filename = '{}/{}_test.svm'.format(data, data)
    x_train, y_train = datasets.load_svmlight_file(train_filename, dtype=np.longdouble, multilabel=True)
    x_test, y_test = datasets.load_svmlight_file(test_filename, dtype=np.longdouble, multilabel=True)
    
    x_train, x_test = np.float32(x_train.A) , np.float32(x_test.A) 
    
    mlb = preprocessing.MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    y_test = mlb.transform(y_test)
    
    return x_train, y_train, x_test, y_test


def test_risk_exact_probit(X, Y, policy):
    
    N_train = len(X)
    idxlist = np.arange(N_train)
    bsize = 128
    risk = 0.
    device = policy.dev
    
    for bnum, st_idx in tqdm(enumerate(range(0, N_train, bsize))):

        end_idx = min(st_idx + bsize, N_train)
        indices = idxlist[st_idx:end_idx]
        
        x, y = X[indices].to(device), Y[indices].to(device)
        bs = x.size(0)
        
        probs_a = policy.policy_a(x, y, n_samples = 10_000).detach()

        risk += -torch.sum(probs_a).item()
    
    return risk/N_train


def test_risk_exact_multiclass(X, Y, policy):
    
    N_train = len(X)
    idxlist = np.arange(N_train)
    bsize = 32
    risk = 0.
    device = policy.dev
    print(device)
    
    for bnum, st_idx in tqdm(enumerate(range(0, N_train, bsize))):

        end_idx = min(st_idx + bsize, N_train)
        indices = idxlist[st_idx:end_idx]
        
        x, y = X[indices].to(device), Y[indices].to(device)
        bs = x.size(0)
        
        probs = policy.policy(x, n_samples = 1000).detach()

        risk += -torch.sum(y * probs).item()
    
    return risk/N_train


def build_bandit_dataset(dataloader, logging_policy, replay_count, multiclass = False, device = torch.device("cpu")):
    logging_policy.eval()
    logging_policy = logging_policy.to(device)
    with torch.no_grad():
        contexts, actions, propensities, costs = [], [], [], []
        for _ in range(replay_count):
            for (contexts_b, y) in tqdm(dataloader) :
                N = len(contexts_b)
                
                contexts_b, y = contexts_b.to(device), y.to(device)
                actions_b = logging_policy.sample(contexts_b).squeeze()
                propensities_b = torch.clip(logging_policy(contexts_b), min=1e-30)
                
                if multiclass :
                    costs_b = -1. * (y[torch.arange(N), actions_b])
                else :
                    costs_b = -1. * (actions_b == y)
                
                contexts.append(contexts_b.cpu())
                actions.append(actions_b.cpu())
                propensities.append(propensities_b.cpu())
                costs.append(costs_b.cpu())
                
        return torch.cat(contexts), torch.cat(actions), torch.cat(propensities), torch.cat(costs)

