"""Multiple sampling from simplex tool."""
import numpy as np


def sample_from_simplices_m_times(p: np.ndarray, m: int) -> np.ndarray:
  """Samples from each of n probability simplices for m times.

  Args:
    p: n-times-K matrix where each row describes a probability simplex
    m: number of times to sample

  Returns:
    n-times-m matrix of indices of simplex corners.
  """
  axis = 1
  r = np.expand_dims(np.random.rand(p.shape[1 - axis], m), axis=axis)
  p_ = np.expand_dims(p.cumsum(axis=axis), axis=2)
  return (np.repeat(p_, m, axis=2) > r).argmax(axis=1)
