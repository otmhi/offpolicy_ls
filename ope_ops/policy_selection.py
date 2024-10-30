from experiments import run_single_experiment
from evaluators import get_estimators
import os
from tabulate import tabulate

import warnings
warnings.filterwarnings("ignore")


estimators = get_estimators(delta = 0.05/4) # we have 4 policies to test with delta 0.05
est_names = [est.get_abbrev() for est in estimators]
result_folder = 'ope_ops/ops_results'

print('The different estimators that are tested are:', est_names)

uci_all = [39, 5, 1233, 11, 1515, 54, 181, 30, 28, 182, 184]

for uci_id in uci_all:
  results, dataset_name, n_log = run_single_experiment(estimators = estimators,
                                                       openml_id = uci_id, 
                                                       reward_noise_p = 0.2,
                                                       n_trials = 10,
                                                       behavior_policy_temperature = 0.2,
                                                       behavior_faulty_actions = [0, 1],
                                                       target_policy_temperature = 0.1)
  csv_name = f'{dataset_name}_{n_log}.csv' 
  print('---------------------------------------------')
  print(f'{dataset_name} with N = {n_log} interactions')
  print('---------------------------------------------')

  print(tabulate(results, headers=results.columns, tablefmt="psql"))
  results.to_csv(os.path.join(result_folder, csv_name), index = False)
