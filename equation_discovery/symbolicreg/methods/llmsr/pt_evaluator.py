

"""
Evaluator for PT-LLMSR that works with the PT Experience Buffer.

Similar to the base evaluator but routes programs to specific pools
based on which pool generated the prompt.
"""
from __future__ import annotations

import ast
import time
import copy
from collections.abc import Sequence
from typing import Any, Type

from llmsr import code_manipulation
from llmsr.pt_buffer import PTExperienceBuffer
from llmsr.evaluator import Sandbox, LocalSandbox, _trim_function_body


def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
    """
    Return the compiled generated function and the full runnable program.
    """
    body = _trim_function_body(generated_code)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            code=body,
            source_name=f'{function_to_evolve}_v{version_generated}',
            target_name=function_to_evolve
        )

    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body

    return evolved_function, str(program)


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """Return whether the generated function is calling an earlier version."""
    for name in code_manipulation.get_functions_called(program):
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False


class PTEvaluator:
    """
    Evaluator for PT-LLMSR that routes programs to appropriate pools.
    
    Extends the base evaluator to track which pool generated the prompt
    and routes new programs back to that pool.
    
    Supports optional OOD (out-of-distribution) evaluation for diversity analysis.
    """

    def __init__(
        self,
        database: PTExperienceBuffer,
        template: code_manipulation.Program,
        function_to_evolve: str,
        function_to_run: str,
        inputs: Sequence[Any],
        timeout_seconds: int = 30,
        sandbox_class: Type[Sandbox] = LocalSandbox,
        ood_inputs: Sequence[Any] = None,
    ):
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._ood_inputs = ood_inputs  # OOD data for diversity analysis
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox_class()

    def analyse(
        self,
        sample: str,
        island_id: int | None,
        version_generated: int | None,
        pool_id: int = 0,
        **kwargs
    ) -> None:
        """
        Compile the hypothesis sample into a program and execute it on test inputs.
        
        Also evaluates on OOD (out-of-distribution) data if available.
        
        Args:
            sample: Generated code sample
            island_id: Target island ID (None for all islands)
            version_generated: Version number of the generated function
            pool_id: Target pool ID within the island
            **kwargs: Additional arguments including profiler
        """
        new_function, program = _sample_to_program(
            sample, version_generated, self._template, self._function_to_evolve
        )
        scores_per_test = {}
        ood_scores_per_test = {}

        time_reset = time.time()

        # Evaluate on training data
        for current_input in self._inputs:
            test_output, runs_ok = self._sandbox.run(
                program,
                self._function_to_run,
                self._function_to_evolve,
                self._inputs,
                current_input,
                self._timeout_seconds
            )

            if (runs_ok and 
                not _calls_ancestor(program, self._function_to_evolve) and 
                test_output is not None):
                if not isinstance(test_output, (int, float)):
                    print(f'Error: test_output is {test_output}')
                    raise ValueError('@function.run did not return an int/float score.')
                scores_per_test[current_input] = test_output

        # Evaluate on OOD data if available
        if self._ood_inputs is not None and scores_per_test:
            for current_input in self._ood_inputs:
                try:
                    test_output, runs_ok = self._sandbox.run(
                        program,
                        self._function_to_run,
                        self._function_to_evolve,
                        self._ood_inputs,
                        current_input,
                        self._timeout_seconds
                    )

                    if (runs_ok and 
                        not _calls_ancestor(program, self._function_to_evolve) and 
                        test_output is not None):
                        if isinstance(test_output, (int, float)):
                            ood_scores_per_test[current_input] = test_output
                except Exception:
                    pass  # OOD evaluation is optional, don't fail on errors

        evaluate_time = time.time() - time_reset

        if scores_per_test:
            self._database.register_program(
                new_function,
                island_id,
                scores_per_test,
                pool_id=pool_id,
                ood_scores_per_test=ood_scores_per_test if ood_scores_per_test else None,
                **kwargs,
                evaluate_time=evaluate_time
            )
        else:
            # Handle failed evaluation for profiling
            profiler = kwargs.get('profiler', None)
            if profiler:
                global_sample_nums = kwargs.get('global_sample_nums', None)
                sample_time = kwargs.get('sample_time', None)
                new_function.global_sample_nums = global_sample_nums
                new_function.score = None
                new_function.sample_time = sample_time
                new_function.evaluate_time = evaluate_time
                profiler.register_function(new_function)


