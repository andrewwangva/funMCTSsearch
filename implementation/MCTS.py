BIN_PACKING_PROMPT = '''\
def priority(item: float, bins: np.ndarray) -> np.ndarray:
  """Returns priority with which we want to add item to each bin.

  Args:
    item: Size of item to be added to the bin.
    bins: Array of capacities for each bin.

  Return:
    Array of same size as bins with priority score of each bin.
  """
'''



class Node:
    def __init__(self, prior, prompt):
        self.prior = prior      # The prior probability of selecting this state from its parent
        self.prompt = prompt
        self.children = {}      # A lookup of legal child positions
        self.visit_count = 0    # Number of times this state was visited during MCTS.
        self.value_sum = 0      # The total value of this state from all visits
        self.state = None       # The board state as this node

    def root_prompt(self):
        self.prompt = BIN_PACKING_PROMPT
    
    def get_prompt(self):
        return self.prompt
    
    def value(self):
        # Average value for a node
        return self.value_sum / self.visit_count
  
  