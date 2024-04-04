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

"""Class for evaluating programs proposed by the Sampler."""
import ast
from collections.abc import Sequence
import copy
from typing import Any
import re

from implementation import code_manipulation
from implementation import programs_database


class _FunctionLineVisitor(ast.NodeVisitor):
	"""Visitor that finds the last line number of a function with a given name."""

	def __init__(self, target_function_name: str) -> None:
		self._target_function_name: str = target_function_name
		self._function_end_line: int | None = None

	def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
		"""Collects the end line number of the target function."""
		if node.name == self._target_function_name:
			self._function_end_line = node.end_lineno
		self.generic_visit(node)

	@property
	def function_end_line(self) -> int:
		"""Line number of the final line of function `target_function_name`."""
		assert self._function_end_line is not None  # Check internal correctness.
		return self._function_end_line


def _normalize_indentation(lines: Sequence[str]) -> list[str]:
	"""Replaces any indentation format with 2 spaces in a list of lines."""
	indentation_used = len(lines[0]) - len(lines[0].lstrip())
	if(indentation_used == 0):
		return lines
	indentation = ' ' * 2
	lines = [(indentation * ((len(line) - len(line.lstrip()))//indentation_used)) + line.lstrip() for line in lines]
	return lines

def _trim_function_body(generated_code: str) -> str:
	"""
	Given code in string form, returns the body of the function "priority" as a string. Any form of tabs are replaced with 4 spaces. 
	The function may come in a couple forms:

	1. The code may be a function with a header
	2. The code may just be the body of a function
	3. The code may include imports

	tests:
	_trim_function_body("  return -(bins - item)")
	returns: "  return -(bins - item)"
	_trim_function_body("def priority():\n  return -(bins - item)")
	returns: "  return -(bins - item)"
	_trim_function_body("import numpy\n\ndef priority():\n  return -(bins - item)")
	returns: "  return -(bins - item)"
	"""
	#find priority
	if("def priority" in generated_code):
		#remove code before priority
		code = generated_code[generated_code.index("def priority"):]
	else:
		code = f'def priority():\n{generated_code}'
	tree = None
	# We keep trying and deleting code from the end until the parser succeeds.
	while tree is None:
		try:
			tree = ast.parse(code)
		except SyntaxError as e:
			code = '\n'.join(code.splitlines()[:e.lineno - 1])
		if not code:
			# Nothing could be saved from `generated_code`
			return ''

	visitor = _FunctionLineVisitor('priority')
	visitor.visit(tree)
	body_lines = code.splitlines()[1:visitor.function_end_line]
	body_lines = _normalize_indentation(body_lines)
	return '\n'.join(body_lines) + '\n\n'


def _sample_to_program(
	generated_code: str,
	version_generated: int | None,
	template: code_manipulation.Program,
	function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
	"""Returns the compiled generated function and the full runnable program."""
	body = _trim_function_body(generated_code)
	if version_generated is not None:
		body = code_manipulation.rename_function_calls(
			body,
			f'{function_to_evolve}_v{version_generated}',
			function_to_evolve)

	program = copy.deepcopy(template)
	evolved_function = program.get_function(function_to_evolve)
	evolved_function.body = body
	return evolved_function, str(program)


class Sandbox:
	"""Sandbox for executing generated code."""
  
	def run(
		self,
		program: str,
		function_to_run: str,
		test_input: Any,
		timeout_seconds: int,
	) -> tuple[Any, bool]:
		"""Returns `function_to_run(test_input)` and whether execution succeeded."""
		try:
			# run the program with decorators funsearch.run and funsearch.evolve
			# remove any decorators from the program
			program = code_manipulation.remove_decorators(program, 'funsearch')
			exec(program, globals())
			#call function to run
			output = globals()[function_to_run](test_input)
			return output, True
		except Exception as e:
			return e, False
		


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
	"""Returns whether the generated function is calling an earlier version."""
	for name in code_manipulation.get_functions_called(program):
		# In `program` passed into this function the most recently generated
		# function has already been renamed to `function_to_evolve` (wihout the
		# suffix). Therefore any function call starting with `function_to_evolve_v`
		# is a call to an ancestor function.
		if name.startswith(f'{function_to_evolve}_v'):
			return True
		return False


class Evaluator:
	"""Class that analyses functions generated by LLMs."""

	def __init__(
		self,
		database: programs_database.ProgramsDatabase,
		template: code_manipulation.Program,
		function_to_evolve: str,
		function_to_run: str,
		inputs: Sequence[Any],
		timeout_seconds: int = 30,
	):
		self._database = database
		self._template = template
		self._function_to_evolve = function_to_evolve
		self._function_to_run = function_to_run
		self._inputs = inputs
		self._timeout_seconds = timeout_seconds
		self._sandbox = Sandbox()

	def analyse(
		self,
		sample: str,
		island_id: int | None,
		version_generated: int | None,
	) -> dict[Any, float]:
		"""Compiles the sample into a program and executes it on test inputs."""
		new_function, program = _sample_to_program(
			sample, version_generated, self._template, self._function_to_evolve)

		scores_per_test = {}
		for name, instances in self._inputs.items():
			test_output, runs_ok = self._sandbox.run(
				program, self._function_to_run, instances, self._timeout_seconds)
			if (runs_ok and not _calls_ancestor(program, self._function_to_evolve)
				and test_output is not None):
				if not isinstance(test_output, (int, float)):
					raise ValueError('@function.run did not return an int/float score.')
				scores_per_test[name] = test_output
			else:
				print("error", test_output)
		return scores_per_test