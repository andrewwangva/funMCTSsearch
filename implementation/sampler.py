# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs."""
from collections.abc import Collection, Sequence

import numpy as np
from openai import OpenAI
import os

from implementation import evaluator
from implementation import MCTS
from implementation import config as config_lib


class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt

  def _draw_strong_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    
    client = OpenAI(
      api_key=os.environ.get("OPENAI_API_KEY"),
    )
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
        temperature=1.2,
    )
    return response.choices[0].message.content.strip()

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    
    client = OpenAI(
      api_key=os.environ.get("OPENAI_API_KEY"),
    )
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
        temperature=1.2,
    )
    return response.choices[0].message.content.strip()

  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]
  def draw_strong_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      tree: MCTS.Node,
      evaluators: Sequence[evaluator.Evaluator],
      samples_per_prompt: int,
      config: config_lib.Config = None,
  ) -> None:
    self.tree = tree
    self._evaluators = evaluators
    self._llm = LLM(samples_per_prompt)
    self.config = config

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""

    min_score = 1000000
    opt_code = None

    for iter in range(1000):
      with open("results.txt", "a") as f:
        f.write(f"iter: {iter}\n")
        f.write("min score: " + str(min_score) + "\n")
        f.write("optimal code: " + str(opt_code) + "\n")
      MCTS.visualize_tree(self.tree)
      print("min score", min_score)
      """
      Select
      """
      selected_node = MCTS.select(self.tree)

      """
      Expansion
      """
      prompt = selected_node.get_prompt()
      samples = self._llm.draw_strong_samples(prompt)
      leaf_nodes = []
      for sample in samples:
        chosen_evaluator = np.random.choice(self._evaluators)
        try:
          scores = chosen_evaluator.analyse(sample, None, None)
          if(scores):
            v = np.exp(1 - (-scores["OR1"] - self.config.opt_num_bins)/self.config.opt_num_bins) #e^(1 - (-score - lower_bound)/lower_bound)
            new_node = selected_node.create_child(evaluator._trim_function_body(sample), prior=v) #change prior
            leaf_nodes.append(new_node)
            if(-scores["OR1"] < min_score):
              min_score = -scores["OR1"]
              opt_code = sample
        except Exception as e:
          pass

      """
      Simulation
      """
      
      for leaf_node in leaf_nodes:
        to_evaluate = []
        for sim in range(5):
          current_node = leaf_node
          for _ in range(3):
            prompt = current_node.get_prompt()
            sample = self._llm._draw_sample(prompt)
            child_node = current_node.create_child(evaluator._trim_function_body(sample))
            current_node = child_node
          to_evaluate.append(current_node)

        """
        Backpropagation
        """
        for node in to_evaluate:
          try:
            scores = chosen_evaluator.analyse(node.parent_action, None, None)
            if(scores):
              v = np.exp(1 - (-scores["OR1"] - self.config.opt_num_bins)/self.config.opt_num_bins)
              MCTS.backprop(node, v) #change reward
              if(-scores["OR1"] < min_score):
                min_score = -scores["OR1"]
                opt_code = sample
          except Exception as e:
            pass
        
        #reset leaf_node children
        leaf_node.children = {}
    return opt_code

  def sample_as_tree(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""

    min_score = 1000000
    opt_code = None

    queue = [self.tree]
    for iter in range(5):
      new_queue = []
      for tree in queue:
        prompt = tree.get_prompt()
        samples = self._llm.draw_samples(prompt)
        # This loop can be executed in parallel on remote evaluator machines.
        for sample in samples:
          chosen_evaluator = np.random.choice(self._evaluators)
          scores = chosen_evaluator.analyse(sample, None, None)
          if(scores):
            new_queue.append(tree.create_child(evaluator._trim_function_body(sample)))
            if(-scores["OR1"] < min_score): #Harcoded in for now
              min_score = -scores["OR1"]
              opt_code = sample
          print("scores: \n", scores)
      queue = new_queue
      print(f"iter: {iter}")
      print(f"min score: {min_score}")
      print(f"optimal code: {opt_code}")


