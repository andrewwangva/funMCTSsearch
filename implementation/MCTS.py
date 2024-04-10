from jinja2 import Template
from graphviz import Digraph
import os
from datetime import datetime

EVOLVE_BIN_PACKING_PROMPT = '''\
Modify the below heuristic function slightly. The function, priority, assigns priority to bins to solve the online binpacking problem. The function should take in a float `item` and a numpy array `bins` and return a numpy array of the same size as `bins` with the priority score of each bin. The priority score should be a float and should be higher for bins with higher priority. The function should not use any external libraries other than `numpy`.
Create a more rich heuristic function by slightly modifying or adding lines. Include the whole function, including the same function signature and the body. Good heuristics might consider factors like the remaining capacity of each bin, the initial capacity of each bin, and the tightness of the packing.

Original function:
{{ Example }}

Modified function:
def priority(item: float, bins: np.ndarray) -> np.ndarray:
'''

INIT_BIN_PACKING_PROMPT = '''\
Write the heuristic function `priority` for online binpacking. The function should take in a float `item` and a numpy array `bins` and return a numpy array of the same size as `bins` with the priority score of each bin. The priority score should be a float and should be higher for bins with higher priority. The function should not use any external libraries other than `numpy`.
Include the whole function, including the function signature and the body. Good heuristics might consider factors like the remaining capacity of each bin, the initial capacity of each bin, and the tightness of the packing.

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
        self.prior = prior      # Replaced with value of current heuristic function
        self.prompt = prompt
        self.children = {}      # A lookup of legal child positions
        self.parent = parent
        self.parent_action = parent_action
        self.visit_count = 0    # Number of times this state was visited during MCTS.
        self.value_sum = 0      # The total value of this state from all visits 
    
    def get_prompt(self):
        return self.prompt
    
    def create_child(self, action, prior=1):
        if action not in self.children.keys():
            template = Template(EVOLVE_BIN_PACKING_PROMPT)
            new_prompt = template.render(Example=action)
            self.children[action] = Node(prior=prior, prompt=new_prompt, parent=self, parent_action=action) #change prior
        return self.children[action]
    
    def selection_value(self):
        u = self.prior/(1 + self.visit_count)
        q = self.prior + self.value_sum/self.visit_count #replace with (lambda * prior) + (1 - lambda) * value_sum/visit_count
        return u + q

    def update(self,reward):
        self.value_sum += reward
        self.visit_count += 1
              
def select(node: Node) -> Node:
    """Selects a child node of `node` by continously getting the child with the highest "select_value" as defined in MCTS."""
    while node.children:
        action = max(node.children, key=lambda n: node.children.get(n).selection_value())
        node = node.children[action]
    return node

def backprop(node: Node, reward: float):
    """Backpropagates the reward from the leaf node to the root node."""
    while node:
        node.update(reward)
        node = node.parent

def visualize_tree(root):
    dot = Digraph(comment='Tree')
    def get_node_label(node):
        # Return the node name or a unique identifier if the name is None
        return f"id-{id(node)}- "+"{:.2f}".format(node.value_sum) if node.value_sum is not None else f"id-{id(node)}"
    def add_nodes_edges(node):
        # Add the current node
        node_label = get_node_label(node)
        dot.node(node_label, node_label)
        
        # Add nodes and edges for children
        for child_name, child_node in node.children.items():
            child_label = get_node_label(child_node)
            dot.node(child_label, child_label)
            dot.edge(node_label, child_label, label=str(child_node.prior))
            add_nodes_edges(child_node)
    
    add_nodes_edges(root)
    os.makedirs("tree_visuals", exist_ok=True)
    
    # Format the current time to include in the filename
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join("tree_visuals", f'tree_{timestamp}')
    
    # Save the graph to a file in PNG format in the specified folder with a timestamp
    dot.render(filename, format='png', view=False)