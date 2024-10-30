from typing import NamedTuple

import numpy as np
import sklearn.datasets as skl_data
import sklearn.model_selection as skl_ms
import sklearn.preprocessing as skl_prep

class FullInfoLoggedData(NamedTuple):
  """A dataset logged by a bandit policy and the true labels for testing.

  Attributes:
    contexts: n-times-d Array -- feature vectors for each entry
    actions: n-times-1 Array -- action taken by logging policy
    p_0: n-times-1 Array -- propensities of the actions under the logging policy
    rewards: n-times-1 Array -- reward received
    labels: n-times-1 Array -- True label
  """

  contexts: np.ndarray
  actions: np.ndarray
  p_0: np.ndarray
  rewards: np.ndarray
  labels: np.ndarray

def generate_binary_noise(
    n: int,
    p: float,
) -> np.ndarray:
  """Returns a Bernoulli-distributed noise vector.

  Args:
    n: Number of points to generate.
    p: Bernoulli parameter (same for each point).
  Returns: Binary vector of length n.
  """
  return np.random.binomial(n=1, p=p, size=n)


def get_reward(
    actions: np.ndarray,
    labels: np.ndarray,
    reward_noise_p: float = 0.1,
    low_rew: float = 0.0,
    high_rew: float = 1.,
):
  """Returns rewards and corrupted labels for matching actions.

  Args:
    actions: A n-vector of actions (integer in {0,nb_class -1}).
    labels: A n-vector of labels.
    reward_noise_p: A noise-level parameter in (0,1).
    low_rew: Reward for incorrect action.
    high_rew: Reward for correct action.
  Returns: A n-vector of rewards after adding noise and rescaling.
  """
  rewards = np.equal(actions, labels)
  rewards = (rewards + generate_binary_noise(rewards.size, reward_noise_p)) % 2
  rewards = high_rew * rewards + low_rew * (1 - rewards)
  return rewards

class Dataset:
  """Represents an OpenML dataset.

  Attributes:
    openml_id: OpenML id of the dataset (for loading).
    train_frac: In (0,1), fraction of the data to train policies.
    log_frac: In (0,1), fraction of the data to generate interactions and evaluate policies.
    reward_noise: In (0,1) noise level in the rewards obtained by a policy.
    name: Name of a dataset according to OpenML.
    encoder: Instance of scikit-learn LabelEncoder() preprocessing labels.

    contexts_all: Full dataset contexts.
    labels_all: Full dataset labels.

    contexts_train: Train data contexts.
    contexts_log: Interaction data context.
    contexts_test: Test data context.

    labels_train: Train data labels.
    labels_log: Interaction data labels.
    labels_test: Test data labels.

    n_train: Train data size.
    n_log: Interaction data size.
    n_test: Test data size.
    size: Total size of the dataset.
  """
  def __init__(
        self,
        openml_id: int,
        standardize: bool = True,
        train_frac: float = 0.20,
        log_frac: float = 0.60,
        random_state: int = 0,
        reward_noise: float = 0.1):
    """Constructs Dataset object.

    Args:
        openml_id: OpenML id of the dataset (for loading).
        standardize: Binary, use True to standardize dataset.
        train_frac: In (0,1), fraction of the data to be used to train the policies.
        log_frac: In (0,1), fraction of the data to be used as interaction data (logged dataset).
        random_state: Seed for train-test split (sklearn).
        reward_noise: In (0,1) noise level in the rewards obtained by a policy.
    """
    self.openml_id = openml_id
    self.train_frac = train_frac
    self.log_frac = log_frac
    self.reward_noise = reward_noise

    dataset = skl_data.fetch_openml(data_id=openml_id, cache=True, as_frame=False)
    data = dataset.data
    target = dataset.target
    self.name = dataset.details["name"]

    self.encoder = skl_prep.LabelEncoder()
    self.encoder.fit(target)
    target = self.encoder.transform(target)

    self.oh_encoder = skl_prep.OneHotEncoder(sparse_output=False)
    self.oh_encoder.fit(target.reshape(-1, 1))

    self.contexts_all = data
    self.labels_all = target

    if standardize:
      scaler = skl_prep.StandardScaler()
      scaler.fit(self.contexts_all)
      self.contexts_all = scaler.transform(self.contexts_all)

    (C_to_use, self.contexts_test, 
     L_to_use, self.labels_test) = skl_ms.train_test_split(self.contexts_all,self.labels_all,
                                                           train_size= self.train_frac + self.log_frac, 
                                                           shuffle=True, random_state=random_state)
    (self.contexts_train, self.contexts_log, 
     self.labels_train, self.labels_log) = skl_ms.train_test_split(C_to_use, L_to_use,
                                                                   train_size= self.train_frac/(self.train_frac + self.log_frac),
                                                                   shuffle=True, random_state=random_state)
    
    self.n_train = len(self.labels_train)
    self.n_log = len(self.labels_log)
    self.n_test = len(self.labels_test)
    self.size = len(self.labels_all)

    self.num_actions = len(self.encoder.classes_)

    del C_to_use, L_to_use


  def get_interaction_logs(self, policy) -> FullInfoLoggedData:
    """Returns logged bandit feedback data: contexts, action, p_0, rewards, test labels.

    Args:
      policy: An object of class Policy
    Returns: A tuple FullInfoLoggedData (contexts, actions, rewards, labels).
    """
    actions, p_0 = policy.query(self.contexts_log)
    rewards = get_reward(actions, self.labels_log, self.reward_noise)


    return FullInfoLoggedData(self.contexts_log, actions, p_0, rewards,
                              self.labels_log)
  

  def get_interaction_logs_for_training(self, policy) -> FullInfoLoggedData:
    """Returns logged bandit feedback data: contexts, action, p_0, rewards, test labels.

    Args:
      policy: An object of class Policy
    Returns: A tuple FullInfoLoggedData (contexts, actions, rewards, labels).
    """
    actions, p_0 = policy.query(self.contexts_train)
    rewards = get_reward(actions, self.labels_train, self.reward_noise)


    return FullInfoLoggedData(self.contexts_train, actions, p_0, rewards,
                              self.labels_train)
  
  def get_true_value(self, policy) -> float:
    """Returns the true value of a policy, approximated with the test set.

    Args:
      policy: An object of class Policy
    Returns: A float (the value of the policy).
    """
    p_vectors = policy.get_probs(self.contexts_test) # N times P
    oh_test_labels = self.oh_encoder.transform(self.labels_test.reshape(-1, 1))

    expected_rewards = (1. - self.reward_noise) * oh_test_labels + self.reward_noise * (1. - oh_test_labels) # N times P
    
    expected_value_x = (p_vectors * expected_rewards).sum(axis = 1) # N times 1
    true_value = expected_value_x.mean()

    return true_value

  def get_action_set(self):
    """Returns dictionary mapping labels to corresponding actions."""

    return self.encoder.transform(self.encoder.classes_)
