from implementation import evaluator
from implementation import funsearch
from bin_packing.bin_packing_data import create_bin_packing_datasets
from implementation import config as config_lib

"""
tests
_trim_function_body("  return -(bins - item)")
returns: "  return -(bins - item)"
_trim_function_body("def priority():\n  return -(bins - item)")
returns: "  return -(bins - item)"
_trim_function_body("import numpy\n\ndef priority():\n  return -(bins - item)")
returns: "  return -(bins - item)"
"""

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
inputs, opt_num_bins = create_bin_packing_datasets()


gen_code = '''\
# Calculate the remaining capacity of each bin after adding the item
  remaining_capacity = bins - item
  
  # Calculate the priority score for each bin
  priority_scores = np.maximum(remaining_capacity, 0) + bins  # Higher priority for bins with more remaining capacity and larger initial capacity
  priority_scores = np.maximum(priority_scores, bins.mean())  # Add a bias towards bins with average capacity
  
  return priority_scores
'''
import pdb; pdb.set_trace()
trim = evaluator._trim_function_body(gen_code)
print("1: ", trim)


generated_code = "bins_priority = np.zeros_like(bins)\n\nfor i in range(len(bins)):\n    if bins[i] >= item:\n        bins_priority[i] = bins[i] - item\n    else:\n        bins_priority[i] = float('inf')\n\nreturn bins_priority"

trim = evaluator._trim_function_body(generated_code)
print("2: ", trim)


generated_code2 = "def priority(item: float, bins: np.ndarray) -> np.ndarray:\n  bins_priority = np.zeros_like(bins)\n  for i in range(len(bins)):\n    if bins[i] >= item:\n      bins_priority[i] = bins[i] - item\n    else:\n      bins_priority[i] = float('inf')\n  return bins_priority\n\n"
trim = evaluator._trim_function_body(generated_code2)
print("3: ", trim)