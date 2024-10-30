import pandas as pd
import warnings
import numpy as np

from tqdm import tqdm

from evaluators import get_estimators
from data import Dataset
from policies import SoftmaxDataPolicy

warnings.filterwarnings("ignore")

np.random.seed(0)

temperatures_logging = [0.3, 0.25, 0.2]
temperatures_targ = [0.1, 0.2, 0.3, 0.4, 0.5]
reward_noises = [0., 0.1, 0.2]
uci_all = [39, 5, 1233, 11, 1515, 54, 181, 30, 28, 182, 184]

np.random.seed(0)


estimators = get_estimators(delta = 0.05)
bounds = estimators[-5:]
bound_names = [bound.get_abbrev() for bound in bounds]

print('Bounds used are:', bound_names)
print('Behavior temperatures used are:', temperatures_logging)
print('Target temperatures used are:', temperatures_targ)
print('reward noises used are:', reward_noises)



values = {bound.get_abbrev(): [] for bound in bounds}
values['dataset'] = []
values['temp_0'] = []
values['temp'] = []

for data_id in tqdm(uci_all):

  for reward_noise in reward_noises:
    
    dataset = Dataset(openml_id = data_id, train_frac= 0.01, log_frac=0.01, reward_noise=reward_noise)
    dataset_name = dataset.name

    for temp_targ in temperatures_targ:

        #Compute the true val
        policy_1 = SoftmaxDataPolicy(dataset = dataset, temperature = temp_targ, faulty_actions = [])
        true_val = dataset.get_true_value(policy_1)

        #Start the experiment
        dataset = Dataset(openml_id = data_id, train_frac = 0.01, log_frac = 0.98, reward_noise=reward_noise)
        idx = np.arange(dataset.n_log) 

        policy_1 = SoftmaxDataPolicy(dataset = dataset, temperature = temp_targ, faulty_actions = [])

  
        for temp in temperatures_logging:
          policy_0 = SoftmaxDataPolicy(dataset = dataset, temperature = temp, faulty_actions = [0, 1])
          contexts, actions, p_0, rewards, _ = dataset.get_interaction_logs(policy_0)

          b_probs = policy_0.get_probs(contexts)
          t_probs = policy_1.get_probs(contexts)

          p_1 = t_probs[idx, actions]

          values['dataset'].append(dataset_name)
          values['temp_0'].append(temp)
          values['temp'].append(temp_targ)

          for bound in bounds:
            bound_name = bound.get_abbrev()
            if  bound_name == 'ESLB':
              bound_value = bound(t_probs, b_probs, actions, rewards)['lower_bound'] 
            else:
              bound_value = bound(p_1, p_0, rewards)['lower_bound']
            norm_radius = (true_val - bound_value)/true_val * (true_val > bound_value)

            values[bound_name].append(norm_radius)


pd_norm_radius = pd.DataFrame(values)

pd_norm_radius.to_csv('ope_results/all_radiuses.csv', index=False)


      
