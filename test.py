from implementation import evaluator
from implementation import funsearch
from bin_packing.bin_packing_data import create_bin_packing_datasets
from implementation import config as config_lib
from implementation.MCTS import Node
import numpy as np

inputs, opt_num_bins = create_bin_packing_datasets()


gen_code = '''\
  # Calculate the remaining capacity of each bin after adding the item
  remaining_capacity = bins - item
  
  # Calculate the priority score for each bin
  priority_scores = np.maximum(remaining_capacity, 0) + bins  # Higher priority for bins with more remaining capacity and larger initial capacity
  priority_scores = np.maximum(priority_scores, bins.mean())  # Add a bias towards bins with average capacity
  
  return priority_scores
'''
trim = evaluator._trim_function_body(gen_code)
print("test 1")
print(trim)

def UCB(nodelist: list[Node], node: Node, c: float) -> float:
  """
  Calculate the UCB1 value of a node
  """
  if node.visit_count == 0:
    return float('inf')
  return node.value() + c * np.sqrt(np.log(nodelist[-1].visit_count) / node.visit_count)



node_list = []
tree  = Node(prior = None)