from implementation import funsearch
from bin_packing.bin_packing_data import create_bin_packing_datasets
from implementation import config as config_lib
import numpy as np
import openai
import os

_PY_PROMPT_EVOLVE_RUN = '''\
"""Finds heuristics for online 1d binpacking."""
import numpy as np

def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
  """Returns indices of bins in which item can fit."""
  return np.nonzero((bins - item) >= 0)[0]


def online_binpack(
    items: tuple[float, ...], bins: np.ndarray
) -> tuple[list[list[float, ...], ...], np.ndarray]:
  """Performs online binpacking of `items` into `bins`."""
  # Track which items are added to each bin.
  packing = [[] for _ in bins]
  # Add items to bins.
  for item in items:
    # Extract bins that have sufficient space to fit item.
    valid_bin_indices = get_valid_bin_indices(item, bins)
    # Score each bin based on heuristic.
    priorities = priority(item, bins[valid_bin_indices])
    # Add item to bin with highest priority.
    best_bin = valid_bin_indices[np.argmax(priorities)]
    bins[best_bin] -= item
    packing[best_bin].append(item)
    # Remove unused bins from packing.
  packing = [bin_items for bin_items in packing if bin_items]
  return packing, bins


@funsearch.run
def evaluate(instances: dict) -> float:
  """Evaluate heuristic function on a set of online binpacking instances."""
  # List storing number of bins used for each instance.
  num_bins = []
  # Perform online binpacking for each instance.
  for name in instances:
    instance = instances[name]
    capacity = instance['capacity']
    items = instance['items']
    # Create num_items bins so there will always be space for all items,
    # regardless of packing order. Array has shape (num_items,).
    bins = np.array([capacity for _ in range(instance['num_items'])])
    # Pack items into bins and return remaining capacity in bins_packed, which
    # has shape (num_items,).
    _, bins_packed = online_binpack(items, bins)
    # If remaining capacity in a bin is equal to initial capacity, then it is
    # unused. Count number of used bins.
    num_bins.append((bins_packed != capacity).sum())
    # Score of heuristic function is negative of average number of bins used
    # across instances (as we want to minimize number of bins).
  return -np.mean(num_bins)


@funsearch.evolve
def priority(item: float, bins: np.ndarray) -> np.ndarray:
  """Returns priority with which we want to add item to each bin.
  
  Args:
  item: Size of item to be added to the bin.
  bins: Array of capacities for each bin.
  
  Return:
  Array of same size as bins with priority score of each bin.
  """
  return -(bins - item)
'''


#inputs = {"u500_00": {'capacity': 150, 'num_items': 500, 'items': [42, 69, 67, 57, 93, 90, 38, 36, 45, 42, 33, 79, 27, 57, 44, 84, 86, 92, 46, 38, 85, 33, 82, 73, 49, 70, 59, 23, 57, 72, 74, 69, 33, 42, 28, 46, 30, 64, 29, 74, 41, 49, 55, 98, 80, 32, 25, 38, 82, 30, 35, 39, 57, 84, 62, 50, 55, 27, 30, 36, 20, 78, 47, 26, 45, 41, 58, 98, 91, 96, 73, 84, 37, 93, 91, 43, 73, 85, 81, 79, 71, 80, 76, 83, 41, 78, 70, 23, 42, 87, 43, 84, 60, 55, 49, 78, 73, 62, 36, 44, 94, 69, 32, 96, 70, 84, 58, 78, 25, 80, 58, 66, 83, 24, 98, 60, 42, 43, 43, 39, 97, 57, 81, 62, 75, 81, 23, 43, 50, 38, 60, 58, 70, 88, 36, 90, 37, 45, 45, 39, 44, 53, 70, 24, 82, 81, 47, 97, 35, 65, 74, 68, 49, 55, 52, 94, 95, 29, 99, 20, 22, 25, 49, 46, 98, 59, 98, 60, 23, 72, 33, 98, 80, 95, 78, 57, 67, 53, 47, 53, 36, 38, 92, 30, 80, 32, 97, 39, 80, 72, 55, 41, 60, 67, 53, 65, 95, 20, 66, 78, 98, 47, 100, 85, 53, 53, 67, 27, 22, 61, 43, 52, 76, 64, 61, 29, 30, 46, 79, 66, 27, 79, 98, 90, 22, 75, 57, 67, 36, 70, 99, 48, 43, 45, 71, 100, 88, 48, 27, 39, 38, 100, 60, 42, 20, 69, 24, 23, 92, 32, 84, 36, 65, 84, 34, 68, 64, 33, 69, 27, 47, 21, 85, 88, 59, 61, 50, 53, 37, 75, 64, 84, 74, 57, 83, 28, 31, 97, 61, 36, 46, 37, 96, 80, 53, 51, 68, 90, 64, 81, 66, 67, 80, 37, 92, 67, 64, 31, 94, 45, 80, 28, 76, 29, 64, 38, 48, 40, 29, 44, 81, 35, 51, 48, 67, 24, 46, 38, 76, 22, 30, 67, 45, 41, 29, 41, 79, 21, 25, 90, 62, 34, 73, 50, 79, 66, 59, 42, 90, 79, 70, 66, 80, 35, 62, 98, 97, 37, 32, 75, 91, 91, 48, 26, 23, 32, 100, 46, 29, 26, 29, 26, 83, 82, 92, 95, 87, 63, 57, 100, 63, 65, 81, 46, 42, 95, 90, 80, 53, 27, 84, 40, 22, 97, 20, 73, 63, 95, 46, 42, 47, 40, 26, 88, 49, 24, 92, 87, 68, 95, 34, 82, 84, 43, 54, 73, 66, 32, 62, 48, 99, 90, 86, 28, 25, 25, 89, 67, 96, 35, 33, 70, 40, 59, 32, 94, 34, 86, 35, 45, 25, 76, 80, 42, 91, 44, 91, 97, 60, 29, 45, 37, 61, 54, 78, 56, 74, 74, 45, 21, 96, 37, 75, 100, 58, 84, 85, 56, 54, 71, 52, 79, 43, 35, 27, 70, 31, 47, 35, 26, 30, 97, 90, 80, 58, 60, 73, 46, 71, 39, 42, 98, 27, 21, 71, 71, 78, 76, 57, 24, 91, 84, 35, 25, 77, 96, 97, 89, 30, 86]}}
inputs = create_bin_packing_datasets()
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Set OPENAI_API_KEY environment variable.")
openai.api_key = api_key
funsearch.main(_PY_PROMPT_EVOLVE_RUN, inputs, config_lib.Config())  # OK
