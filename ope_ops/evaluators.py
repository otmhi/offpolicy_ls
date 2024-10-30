import abc
import numpy as np
import math
from typing import List
from absl import logging

import policies
import data

import utils


class Estimator(abc.ABC):
  """Abstract class for a value estimator."""

  @abc.abstractmethod
  def get_name(self) -> str:
    """Returns a the name of the estimator.
    """

  @abc.abstractmethod
  def get_abbrev(self) -> str:
    """Returns a shorter version of the name returned by get_name.
    """

# ==============================================================================
# ==============================================================================
# ==============================================================================
#         Some Classical Estimators.
# ==============================================================================
# ==============================================================================
# ==============================================================================

class IPSEstimator(Estimator):
  """Implements an importance-weighted estimator of the value."""

  def __call__(
      self,
      p_1: np.ndarray,
      p_0: np.ndarray,
      rewards: np.ndarray,
  ):
    """Computes an importance-weighted/Inverse Propensity Scoring (IPS) estimate.

    Args:
      p_1: n-times-1 matrix, π(A_i | X_i)
      p_0: n-times-1 matrix, π_0(A_i | X_i)

      rewards: n-sized reward vector.

    Returns:
      A dictionary with 1 entry:
        estimate: Importance-weighted estimate (required by select_policy(...)).
    """
    n = len(rewards)

    # Importance weights
    weights = p_1 / p_0

    estimate = rewards.dot(weights) / n

    return dict(estimate=estimate)

  def get_name(self):
    """Returns a long name of an estimator."""
    return "Importance-weighted estimator"

  def get_abbrev(self):
    """Returns a short name of an estimator."""
    return "IPS"
  

class ClippedIPSEstimator(Estimator):
  """Implements Clipped IPS."""

  def __init__(self, M:float) -> None:
    super().__init__()
    self.M = M

  def __call__(
      self,
      p_1: np.ndarray,
      p_0: np.ndarray,
      rewards: np.ndarray,
  ):
    """Computes an importance-weighted/Inverse Propensity Scoring (IPS) estimate.

    Args:
      p_1: n-times-1 matrix, π(A_i | X_i)
      p_0: n-times-1 matrix, π_0(A_i | X_i)

      rewards: n-sized reward vector.

    Returns:
      A dictionary with 1 entry:
        estimate: Importance-weighted estimate (required by select_policy(...)).
    """
    n = len(rewards)

    # Importance weights
    weights = np.minimum(p_1 / p_0, self.M)

    estimate = rewards.dot(weights) / n

    return dict(estimate=estimate)

  def get_name(self):
    """Returns a long name of an estimator."""
    return "Clipped IPS estimator"

  def get_abbrev(self):
    """Returns a short name of an estimator."""
    return "CIPS"
  

class IXEstimator(Estimator):
  """Implements Implicit Exploration IPS."""

  def __init__(self, M:float) -> None:
    super().__init__()
    self.M = M

  def __call__(
      self,
      p_1: np.ndarray,
      p_0: np.ndarray,
      rewards: np.ndarray,
  ):
    """Computes an importance-weighted/Inverse Propensity Scoring (IPS) estimate.

    Args:
      p_1: n-times-1 matrix, π(A_i | X_i)
      p_0: n-times-1 matrix, π_0(A_i | X_i)

      rewards: n-sized reward vector.

    Returns:
      A dictionary with 1 entry:
        estimate: Importance-weighted estimate (required by select_policy(...)).
    """
    n = len(rewards)

    # Importance weights
    weights = p_1 / (p_0 + 1/self.M)

    estimate = rewards.dot(weights) / n

    return dict(estimate=estimate)

  def get_name(self):
    """Returns a long name of an estimator."""
    return "Implicit Exploration IPS estimator"

  def get_abbrev(self):
    """Returns a short name of an estimator."""
    return "IX-IPS"
  


class LogEstimator(Estimator):
  """Implements Log IPS."""

  def __init__(self, M:float) -> None:
    super().__init__()
    self.M = M

  def __call__(
      self,
      p_1: np.ndarray,
      p_0: np.ndarray,
      rewards: np.ndarray,
  ):
    """Computes an importance-weighted/Inverse Propensity Scoring (IPS) estimate.

    Args:
      p_1: n-times-1 matrix, π(A_i | X_i)
      p_0: n-times-1 matrix, π_0(A_i | X_i)

      rewards: n-sized reward vector.

    Returns:
      A dictionary with 1 entry:
        estimate: Importance-weighted estimate (required by select_policy(...)).
    """
    n = len(rewards)

    # Importance weights
    x = rewards * (p_1 / p_0 ).ravel()

    estimate = (self.M * np.log(1 + x/self.M) ).mean() 

    return dict(estimate=estimate)

  def get_name(self):
    """Returns a long name of an estimator."""
    return "Log IPS estimator"

  def get_abbrev(self):
    """Returns a short name of an estimator."""
    return "Log-IPS"



class SNIWEstimator(Estimator):
  """Implements Self Normalized Importance Weighting."""

  def __init__(self) -> None:
    super().__init__()

  def __call__(
      self,
      p_1: np.ndarray,
      p_0: np.ndarray,
      rewards: np.ndarray,
  ):
    """Computes aSelf Normalized Importance Weighting estimate.

    Args:
      p_1: n-times-1 matrix, π(A_i | X_i)
      p_0: n-times-1 matrix, π_0(A_i | X_i)

      rewards: n-sized reward vector.

    Returns:
      A dictionary with 1 entry:
        estimate: Self Normalized Importance Weighting estimate (required by select_policy(...)).
    """
    n = len(rewards)

    # Importance weights
    weights = p_1 / p_0

    estimate = rewards.dot(weights) / weights.sum()

    return dict(estimate=estimate)

  def get_name(self):
    """Returns a long name of an estimator."""
    return "Self Normalized Importance Weighting estimator"

  def get_abbrev(self):
    """Returns a short name of an estimator."""
    return "SNIW"
  


# ==============================================================================
# ==============================================================================
# ==============================================================================
#         Finite-Sample lower bounds.
# ==============================================================================
# ==============================================================================
# ==============================================================================
  
  
  
class CIPSEmpBernsteinEstimator(Estimator):
  """Implements an empirical Bernstein lower bound for Clipped IPS.
  Attributes:
    delta: Error probability in (0,1).
  """

  def __init__(self, delta: float):
    """Constructs an lower bound.

    The lower bound holds with probability 1-delta.
    Args:
      delta: Error probability in (0,1).
    """
    self.delta = delta

  def get_name(self):
    """Returns a long name of an estimator."""
    return ("Empirical Bernstein bound CIPS (M=sqrt(n))")

  def get_abbrev(self):
    """Returns a short name of an estimator."""
    return "Emp. Bernstein for CIPS"

  def __call__(
      self,
      p_1: np.ndarray,
      p_0: np.ndarray,
      rewards: np.ndarray,
      M: float = None,
  ):
    """Computes Empirical Bernstein lower bound for CIPS estimate.

    Here n is a sample size.

    Args:
      p_1: n-times-1 matrix, π(A_i | X_i)
      p_0: n-times-1 matrix, π_0(A_i | X_i)

      rewards: n-sized reward vector.

    Returns:
      A dictionary with 5 entries:
        lower_bound: Corresponds to the actual lower bound.
        estimate: Same as lower_bound (required by select_policy(...)).
        est_value: Empirical value.
        concentration: Concentration term.
        est_var: sample variance of the estimator.
    """
    n = len(rewards)

    conf = math.log(2.0 / self.delta)
    if M is None: 
      M = math.sqrt(n)

    # Clipped Importance weights
    weights_M = np.minimum(p_1/p_0, M)

    v_estimates = weights_M * rewards

    est_value = np.mean(v_estimates)
    est_var = np.var(v_estimates)

    concentration = math.sqrt((2 * conf / n) * est_var) + M * (7 * conf) / (3 * (n - 1))

    lower_bound = est_value - concentration

    return dict(
        estimate= max(0, lower_bound),
        lower_bound= max(0, lower_bound),
        est_value=est_value,
        concentration=concentration,
        est_var=est_var)


class CIPSEmpSecondMomentEstimator(Estimator):
  """Implements an empirical Second Moment lower bound for Clipped IPS.
  This bound holds uniformly for all values of M.
  Attributes:
    delta: Error probability in (0,1).
  """

  def __init__(self, delta: float):
    """Constructs an lower bound.

    The lower bound holds with probability 1-delta.
    Args:
      delta: Error probability in (0,1).
    """
    self.delta = delta

  def get_name(self):
    """Returns a long name of an estimator."""
    return ("Empirical Second Moment bound for CIPS")

  def get_abbrev(self):
    """Returns a short name of an estimator."""
    return "L = 1 Emp. SM for CIPS"

  def __call__(
      self,
      p_1: np.ndarray,
      p_0: np.ndarray,
      rewards: np.ndarray,

  ):
    """Computes Empirical Second Moment lower bound for CIPS estimate.

    Here n is a sample size.

    Args:
      p_1: n-times-1 matrix, π(A_i | X_i)
      p_0: n-times-1 matrix, π_0(A_i | X_i)

      rewards: n-sized reward vector.

    Returns:
      A dictionary with 5 entries:
        lower_bound: Corresponds to the actual lower bound.
        estimate: Same as lower_bound (required by select_policy(...)).
        est_value: Empirical value.
        concentration: Concentration term.
        est_var: sample second moment of the estimator.
    """
    n = len(rewards)

    conf = math.log(1.0 / self.delta)
    
    M = math.sqrt(n)
    lambd = math.sqrt(n)

    # Clipped Importance weights
    w = p_1/p_0
    weights_M = np.minimum(w, M)


    v_estimates = weights_M * rewards

    est_value = np.mean(v_estimates)
    est_sm = np.mean(v_estimates**2)

    concentration = est_sm/(2. * lambd) + lambd * conf/n

    to_contract = est_value - concentration

    lower_bound = lambd * (math.exp( to_contract/lambd ) - 1.)

    return dict(
        estimate= max(0, lower_bound),
        lower_bound=  max(0, lower_bound),
        est_value=est_value,
        concentration=concentration,
        est_sm=est_sm)

    

class IXBoundEstimator(Estimator):
  """Implements the Implicit Exploration lower bound for IX-IPS.
  Attributes:
    delta: Error probability in (0,1).
  """

  def __init__(self, delta: float):
    """Constructs a lower bound.

    The lower bound holds with probability 1-delta.
    Args:
      delta: Error probability in (0,1).
    """
    self.delta = delta

  def get_name(self):
    """Returns a long name of an estimator."""
    return ("IX bound")

  def get_abbrev(self):
    """Returns a short name of an estimator."""
    return "Implicit Exploration Bound"

  def __call__(
      self,
      p_1: np.ndarray,
      p_0: np.ndarray,
      rewards: np.ndarray,
      lambd: float = None,
  ):
    """Computes Implicit Exploration Bound.

    Here n is a sample size.

    Args:
      p_1: n-times-1 matrix, π(A_i | X_i)
      p_0: n-times-1 matrix, π_0(A_i | X_i)

      rewards: n-sized reward vector.

    Returns:
      A dictionary with 4 entries:
        lower_bound: Corresponds to the actual lower bound.
        estimate: Same as lower_bound (required by select_policy(...)).
        est_value: Empirical value.
        concentration: Concentration term.
    """
    n = len(rewards)

    conf = math.log(1.0 / self.delta)
    
    if lambd is None:
      lambd = 2 * math.sqrt(n)

    ix_w = p_1/(p_0 + 1/lambd)
    ix_v =  ix_w * rewards

    est_value = np.mean(ix_v)

    concentration =  lambd * conf / (2 * n)

    lower_bound = est_value - concentration

    return dict(
        estimate= max(0, lower_bound),
        lower_bound= max(0, lower_bound),
        est_value=est_value,
        concentration=concentration)
  


class PsiBoundEstimator(Estimator):
  """Implements the Psi lower bound for IPS.
  Attributes:
    delta: Error probability in (0,1).
  """

  def __init__(self, delta: float):
    """Constructs a lower bound.

    The lower bound holds with probability 1-delta.
    Args:
      delta: Error probability in (0,1).
    """
    self.delta = delta

  def get_name(self):
    """Returns a long name of an estimator."""
    return ("Psi bound")

  def get_abbrev(self):
    """Returns a short name of an estimator."""
    return "LS Bound"

  def __call__(
      self,
      p_1: np.ndarray,
      p_0: np.ndarray,
      rewards: np.ndarray,
  ):
    """Computes Psi bound for IPS estimate.

    Here n is a sample size.

    Args:
      p_1: n-times-1 matrix, π(A_i | X_i)
      p_0: n-times-1 matrix, π_0(A_i | X_i)

      rewards: n-sized reward vector.

    Returns:
      A dictionary with 4 entries:
        lower_bound: Corresponds to the actual lower bound.
        estimate: Same as lower_bound (required by select_policy(...)).
        est_value: Empirical value.
        concentration: Concentration term.
    """
    n = len(rewards)

    conf = math.log(1.0 / self.delta)
    

    # Importance weights
    w = p_1/p_0
    v_estimates = w * rewards

    lambd = math.sqrt(n)

    psi_v = lambd * np.log(1 + v_estimates/lambd)

    est_value = np.mean(psi_v)

    concentration =  lambd * conf / n
    to_contract = est_value - concentration

    lower_bound = lambd * (math.exp( to_contract/lambd ) - 1.)

    return dict(
        estimate= max(0, lower_bound),
        lower_bound= max(0, lower_bound),
        est_value=est_value,
        concentration=concentration)
  



class ESLB(Estimator):
  """Implements a Semi-Empirical Efron-Stein bound for the SNIW (Self-normalized Importance Weighted estimator).

  Attributes:
    delta: Error probability in (0,1).
    n_iterations: Number of Monte-Carlo simulation iterations for approximating
      a multiplicative bias and a variance proxy.
    n_batch_size: Monte-Carlo simulation batch size.
    bias_type: type of bias control to use (see ESLBBiasType).
  """

  def __init__(
      self,
      delta: float,
      n_iterations: int,
      n_batch_size: int,
      bias_type: str
  ):
    """Constructs an estimator.

    The estimate holds with probability 1-delta.

    Args:
      delta: delta: Error probability in (0,1) for a confidence interval.
      n_iterations: Monte-Carlo simulation iterations.
      n_batch_size: Monte-Carlo simulation batch size.
      bias_type: type of bias control to use.
    """
    self.delta = delta
    self.n_iterations = n_iterations
    self.n_batch_size = n_batch_size
    self.bias_type = bias_type

  def get_name(self):
    """Returns the long name of the estimator."""
    return "Semi-Empirical Efron-Stein bound for the Self-normalized Estimator"

  def get_abbrev(self):
    """Returns the short name of the estimator."""
    return "ESLB"

  def __call__(
      self,
      t_probs: np.ndarray,
      b_probs: np.ndarray,
      actions: np.ndarray,
      rewards: np.ndarray,
  ):
    """Computes Efron-Stein lower bound of Theorem 1 as described in Algorithm 1.

    Here n is a sample size, while K is a number actions.

    Args:
      t_probs: n-times-K matrix, where i-th row corresponds to π_t(. | X_i)
        (target probabilities under the target policy).
      b_probs: n-times-K matrix, where i-th row corresponds to π_b(. | X_i)
        (target probabilities under the behavior policy).
      actions: n-sized vector of actions.
      rewards: n-sized reward vector.

    Returns:
      A dictionary with 8 entries:
        lower_bound: Corresponds to the actual lower bound.
        estimate: Same as lower_bound (required by select_policy(...)).
        est_value: Empirical value.
        mult_bias: Multiplicative bias.
        concentration_of_contexts: Hoeffding term, concentration of contexts.
        var_proxy: Variance proxy.
        expected_variance_proxy: Estimated expected counterpart.
    """
    conf = math.log(2.0 / self.delta)
    n = len(actions)
    ix_1_n = np.arange(n)

    # Importance weights
    weights = t_probs[ix_1_n, actions] / b_probs[ix_1_n, actions]

    weights_cumsum = weights.cumsum()
    weights_cumsum = np.repeat(
        np.expand_dims(weights_cumsum, axis=1), self.n_batch_size, axis=1)
    weights_repeated = np.repeat(
        np.expand_dims(weights, axis=1), self.n_batch_size, axis=1)

    weight_table = t_probs / b_probs

    var_proxy_unsumed = np.zeros((n,))
    expected_var_proxy_unsumed = np.zeros((n,))
    loo_expected_recip_weights = 0.0

    are_rewards_binary = ((rewards==0) | (rewards==1)).all()
    if self.bias_type == "MultOneHot" and not are_rewards_binary:
      raise Exception("""bias_type=MultOneHot only supports one-hot rewards. Consider using bias_type=Bernstein""")

    logging.debug(
        "ESLB:: Running Monte-Carlo estimation of the variance proxy and bias")
    logging.debug("ESLB:: iterations = %d, batch size = %d", self.n_iterations,
                  self.n_batch_size)

    for i in range(self.n_iterations):
      actions_sampled = utils.sample_from_simplices_m_times(
          b_probs, self.n_batch_size)
      weights_sampled = weight_table[ix_1_n, actions_sampled.T].T
      weights_sampled_cumsum = weights_sampled[::-1, :].cumsum(axis=0)[::-1, :]
      # Hybrid sums: sums of empirical and sampled weights
      weights_hybrid_sums = np.copy(weights_cumsum)
      weights_hybrid_sums[:-1, :] += weights_sampled_cumsum[1:, :]

      # Computing variance proxy
      weights_hybrid_sums_replace_k = weights_hybrid_sums - weights_repeated + weights_sampled

      sn_weights = weights_repeated / weights_hybrid_sums
      sn_weights_prime = weights_sampled / weights_hybrid_sums_replace_k

      var_proxy_t = (sn_weights + sn_weights_prime)**2
      var_proxy_new_item = var_proxy_t.mean(axis=1)
      var_proxy_unsumed += (var_proxy_new_item - var_proxy_unsumed) / (i + 1)

      actions_sampled_for_expected_var = utils.sample_from_simplices_m_times(
          b_probs, self.n_batch_size)
      weights_sampled_for_expected_var = weight_table[
          ix_1_n, actions_sampled_for_expected_var.T].T

      expected_var_proxy_new_item = (
          (weights_sampled_for_expected_var /
           weights_sampled_for_expected_var.sum(axis=0))**2).mean(axis=1)
      expected_var_proxy_unsumed += (expected_var_proxy_new_item -
                                     expected_var_proxy_unsumed) / (i + 1)


      if self.bias_type == "MultOneHot":
        # Computing bias (loo = leave-one-out)
        # Rewards are `one-hot'
        actions_sampled_for_bias = utils.sample_from_simplices_m_times(
          b_probs, self.n_batch_size)
        weights_sampled_for_bias = weight_table[ix_1_n,
                                                actions_sampled_for_bias.T].T
        loo_sum_weights = np.outer(
          np.ones((n,)),
          np.sum(weights_sampled_for_bias, axis=0)
        ) - weights_sampled_for_bias
        loo_expected_recip_weights += (1 / np.min(loo_sum_weights, axis=0)).mean()

    var_proxy = var_proxy_unsumed.sum()
    expected_var_proxy = expected_var_proxy_unsumed.sum()

    if self.bias_type == "MultOneHot":
      loo_expected_recip_weights /= self.n_iterations
      eff_sample_size = 1.0 / loo_expected_recip_weights
      mult_bias = min(1.0, eff_sample_size / n)
      add_bias = 0
    elif self.bias_type == "Bernstein":
      # Computing Bernstein bias control (based on lower tail Bernstein's inequality)
      expected_sum_weights_sq = (t_probs**2 / b_probs).sum()
      bias_x = math.log(n) / 2
      mult_bias = 1 - math.sqrt(2 * expected_sum_weights_sq * bias_x) / n
      mult_bias = max(0, mult_bias)
      add_bias = math.exp(-bias_x)
      
    concentration = math.sqrt(
        2.0 * (var_proxy + expected_var_proxy) *
        (conf + 0.5 * math.log(1 + var_proxy / expected_var_proxy)))
    concentration_of_contexts = math.sqrt(conf / (2 * n))
    est_value = weights.dot(rewards) / weights.sum()
    lower_bound = mult_bias * (est_value
                               - concentration
                               - add_bias) - concentration_of_contexts

    return dict(
        estimate= max(0, lower_bound),
        lower_bound= max(0, lower_bound),
        est_value=est_value,
        concentration=concentration,
        mult_bias=mult_bias,
        concentration_of_contexts=concentration_of_contexts,
        var_proxy=var_proxy,
        expected_var_proxy=expected_var_proxy)


# ==============================================================================
# ==============================================================================
# ==============================================================================
#         Evaluation utilities for experiments.
# ==============================================================================
# ==============================================================================
# ==============================================================================


def select_policy(
    interaction_data: data.FullInfoLoggedData,
    behavior_policy: policies.Policy,
    t_policies: List[policies.Policy],
    estimator: Estimator,
):
  """Selects a policy given an estimator.

  Args:
    contexts: A n x d matrix of n context vectors.
    actions: A n-vector of actions.
    rewards: A n-vector of rewards.
    b_policy: Behavior policy implementing get_probs(...) method (see
      SoftmaxDataPolicy in policies.py).
    t_policies: A list of objects of implementing get_probs(...) method
      (see SoftmaxGAPolicy).
    estimator: An object of a base class Estimator.

  Returns:
    A tuple (estimate, policy) with the highest estimate.
  """

  estimates_and_policies = []
  contexts, actions, p_0, rewards, _ = interaction_data
  n = len(rewards)
  idx = np.arange(n)

  b_probs = behavior_policy.get_probs(contexts)

  for pol in t_policies:
    t_probs = pol.get_probs(contexts)
    p_1 = t_probs[idx, actions]

    if estimator.get_abbrev() == 'ESLB':
      result_dict = estimator(b_probs = b_probs, t_probs = t_probs, actions = actions, rewards=rewards)
    else:
      result_dict = estimator(p_1 = p_1, p_0 = p_0, rewards=rewards)
    estimates_and_policies.append((result_dict["estimate"], pol))

  ordered_estimates_and_policies = sorted(estimates_and_policies, key=lambda x: x[0])

  return ordered_estimates_and_policies[-1]


def evaluate_estimators(
    interaction_data: data.FullInfoLoggedData,
    behavior_policy: policies.Policy,
    t_policies: List[policies.Policy],
    estimators: List[Estimator],
    dataset: data.Dataset,
):
  """Evaluates multiple estimators based on their ability to select a policy.

  Args:
    interaction_data: A FullInfoLoggedData NamedTuple
    t_policies: A list of n_pol objects of implementing get_probs(...) method.
    estimators: A list of n_est objects of a base class Estimator.
    dataset: Object of the class Dataset.

  Returns:
    A tuple with three elements: (true_value_winners, winners, all_pol_true_values)
  """

  winners_for_est = []  # winner policies of each estimator
  true_value_winners = [] # true value of each winner policy

  for (est_i, est) in enumerate(estimators):
    est_winner, pol_winner = select_policy(interaction_data, behavior_policy, t_policies, est)
    winners_for_est.append((pol_winner, est.get_abbrev()))

    pol_winner_true_value = dataset.get_true_value(pol_winner)
    true_value_winners.append((pol_winner_true_value, est.get_abbrev(), est_winner))

    # Getting test reward of the best policy (as a reference)
    all_pol_true_values = []
    for pol in t_policies:
      true_value_for_pol = dataset.get_true_value(pol)
      all_pol_true_values.append((true_value_for_pol, pol))
    all_pol_true_values = sorted(all_pol_true_values, key=lambda x: x[0])

  return winners_for_est, true_value_winners, all_pol_true_values


def interpret_results(value:float, 
                      p_0_true_value:float, 
                      p_star_true_value:float):
  """""
  Function to compare the policy selected by our estimator to p_0 and the best policy.
  """
  if value == p_star_true_value: return 'BEST'
  if value >= p_0_true_value: return 'BETTER'
  return 'WORSE'


def get_estimators(delta):
  """Constructs estimators to be used in the benchmark.

  Args:
    M: Clipping factor.
    delta: Error probability in (0,1).

  Returns:
    A list of dictionaries containing at least one entry "estimate" (key).

  """
  estimators = [
      IPSEstimator(),
      SNIWEstimator(),
      CIPSEmpBernsteinEstimator(delta=delta),
      ESLB(delta = delta, n_iterations = 10, n_batch_size = 1000, bias_type="Bernstein"),
      IXBoundEstimator(delta=delta),
      CIPSEmpSecondMomentEstimator(delta=delta),
      PsiBoundEstimator(delta=delta)
  ]
  return estimators
