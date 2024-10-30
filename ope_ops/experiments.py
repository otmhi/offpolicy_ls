from typing import List, Tuple
import numpy as np
from evaluators import Estimator, evaluate_estimators, interpret_results
from absl import logging

import data
import policies
import pandas as pd



def run_single_experiment(
    estimators: List[Estimator],
    openml_id: int,
    reward_noise_p: float,
    n_trials: int,
    behavior_policy_temperature: float,
    behavior_faulty_actions: List[int],
    target_policy_temperature:float,
) -> Tuple[np.ndarray, np.ndarray, int]:
  """Returns scores of an experiment on a single dataset for all estimators.

  Evaluates all estimators on a single dataset for a given number of trials,
  given description of the behavior policy and specifications of target
  policies.

  Args:
    estimators: A list of objects of a base class Estimator (imported as est).
    openml_id: OpenML dataset id (integer).
    n_trials: Number of experimental trials (data splits).
    behavior_policy_temperature: Positive float controlling the temperature of a
      Softmax behavior policy.
    behavior_faulty_actions: List of labels on which the behavior policy makes
      mistakes.
    target_policy_specs: Tuple of target policy specifications consisting of
      two-element
    tuples("<policy class name>", <dict of arguments to be passed to the
      constructor>) e.g. ( ("SoftmaxGAPolicy", dict( step_size=0.1, steps=1000,
      temperature=0.1, obj_type=policies.TrainedPolicyObjType.IW)), ).
    reward_noise_p: Probability of a Bernoulli noise added to the reward.

  Returns:
    Tuple (all_test_rewards, all_reference_rewards, dataset_name).

    Here all_test_rewards is np.array
    of dimension (#estimators, #datasets, n_trials) where each entry is a
    reward of a given estimator on a dataset at a particular trial (data split).

    all_reference_rewards is is np.array
    of dimension (#datasets, n_trials) where each entry is a
    reward of a best estimator in a hindsight on a dataset at a particular
    trial (data split).

    dataset_name is a human-readable OpenML dataset name.
  """

  np.random.seed(1)

  dataset = data.Dataset(
      openml_id,
      train_frac= 0.2,
      log_frac=0.5,
      reward_noise=reward_noise_p,
      random_state=0)
  
  est_names = [est.get_abbrev() for est in estimators]
  results = pd.DataFrame(data = np.zeros((n_trials, len(estimators))), columns = est_names, dtype=str)

  action_set = dataset.get_action_set()

  behavior_policy = policies.SoftmaxDataPolicy(dataset = dataset, 
                                               temperature = behavior_policy_temperature, 
                                               faulty_actions = behavior_faulty_actions)
  
  perfect_policy = policies.SoftmaxDataPolicy(dataset, .1, [])
  
  train_interaction_data = dataset.get_interaction_logs_for_training(behavior_policy)

  c, a, p_0, r, _ = train_interaction_data

  IW_pol = policies.SoftmaxGAPolicy(action_set = action_set, temperature = 1., obj_type= 'IW', steps = 50_000)
  IW_pol.train(c, a, r, p_0)
  IW_pol.temperature = target_policy_temperature

  SNIW_pol = policies.SoftmaxGAPolicy(action_set = action_set, temperature = 1., obj_type= 'SNIW', steps = 50_000)
  SNIW_pol.train(c, a, r, p_0)
  SNIW_pol.temperature = target_policy_temperature

  list_policies = [perfect_policy, IW_pol, SNIW_pol, behavior_policy]

  p_0_true_value = dataset.get_true_value(behavior_policy)
  p_star_true_value = dataset.get_true_value(perfect_policy)

  for i_trial in range(n_trials):
    logging.info(
        "\u001b[32mrun_single_experiment:: trial = %d/%d ... \u001b[0m",
        i_trial + 1, n_trials)
    np.random.seed(i_trial)

    interaction_data = dataset.get_interaction_logs(behavior_policy)

    _, true_value_winners, _ = evaluate_estimators(interaction_data = interaction_data, 
                                                   behavior_policy=behavior_policy,
                                                   t_policies = list_policies, 
                                                   estimators = estimators, dataset = dataset)
    
    for result_tuple in true_value_winners:
      v_est, name_est, _ = result_tuple
      results[name_est][i_trial] = interpret_results(v_est, p_0_true_value, p_star_true_value)
    
  return results, dataset.name, dataset.n_log

