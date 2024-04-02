EVOLVE_BIN_PACKING_PROMPT = '''\
Modify the below heuristic function `priority` for online binpacking. The function should take in a float `item` and a numpy array `bins` and return a numpy array of the same size as `bins` with the priority score of each bin. The priority score should be a float and should be higher for bins with higher priority. The function should not use any external libraries other than `numpy`.
Create a more complex heuristic function.

"""
Returns priority with which we want to add item to each bin.

Args:
  item: Size of item to be added to the bin.
  bins: Array of capacities for each bin.

Return:
  Array of same size as bins with priority score of each bin.
"""
def priority(item: float, bins: np.ndarray) -> np.ndarray:
'''

INIT_BIN_PACKING_PROMPT = '''\
Write the heuristic function `priority` for online binpacking. The function should take in a float `item` and a numpy array `bins` and return a numpy array of the same size as `bins` with the priority score of each bin. The priority score should be a float and should be higher for bins with higher priority. The function should not use any external libraries other than `numpy`.
"""
Returns priority with which we want to add item to each bin.

Args:
  item: Size of item to be added to the bin.
  bins: Array of capacities for each bin.

Return:
  Array of same size as bins with priority score of each bin.
"""
def priority(item: float, bins: np.ndarray) -> np.ndarray:
'''

class Node:
    def __init__(self, prior, prompt = INIT_BIN_PACKING_PROMPT, parent = None, parent_action = None):
        self.prior = prior      # The prior probability of selecting this state from its parent
        self.prompt = prompt
        self.children = {}      # A lookup of legal child positions
        self.parent = parent
        self.parent_action = parent_action
        self.visit_count = 0    # Number of times this state was visited during MCTS.
        self.value_sum = 0      # The total value of this state from all visits
    
    def get_prompt(self):
        return self.prompt
    
    def create_child(self, action):
        if action not in self.children.keys():
            self.children[action] = Node(prior=1, prompt=EVOLVE_BIN_PACKING_PROMPT + action, parent=self, parent_action=action)
        return self.children[action]

    def value(self):
        # Average value for a node
        return self.value_sum / self.visit_count
  
  