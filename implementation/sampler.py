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


class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt

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
        temperature=1,
    )
    print("response :\n", response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()

  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      tree: MCTS.Node,
      evaluators: Sequence[evaluator.Evaluator],
      samples_per_prompt: int,
  ) -> None:
    self.tree = tree
    self._evaluators = evaluators
    self._llm = LLM(samples_per_prompt)

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""

    min_score = 1000000
    opt_code = None

    queue = [self.tree]
    for _ in range(5):
      new_queue = []
      for tree in queue:
        prompt = tree.get_prompt()
        sample = self._llm.draw_samples(prompt)[0]
        chosen_evaluator = np.random.choice(self._evaluators)
        scores = chosen_evaluator.analyse(sample, None, None)
        
        if(scores):
          new_queue.append(tree.create_child(evaluator._trim_function_body(sample)))
          if(-scores["OR1"] < min_score):
            min_score = -scores["OR1"]
            opt_code = sample
        print("scores: \n", scores)
      queue = new_queue
      print(min_score)
      print(opt_code)


