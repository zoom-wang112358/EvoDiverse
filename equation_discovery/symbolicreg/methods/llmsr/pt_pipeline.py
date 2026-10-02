

"""
Implementation of the PT-LLMSR pipeline.

This extends the base LLMSR pipeline with Parallel Tempering (PT) capabilities,
enabling diversity-preserving evolutionary search through temperature-based
pool swapping.
"""
from __future__ import annotations

from typing import Any, Tuple, Sequence

from llmsr import code_manipulation
from llmsr import config as config_lib
from llmsr import profile
from llmsr.pt_buffer import PTExperienceBuffer
from llmsr.pt_evaluator import PTEvaluator
from llmsr.pt_sampler import PTSampler


def _extract_function_names(specification: str) -> Tuple[str, str]:
    """Return the name of the function to evolve and of the function to run.

    The specification MUST have two functions decorated with 
    '@evaluate.run' and '@equation.evolve' respectively.
    """
    run_functions = list(code_manipulation.yield_decorated(specification, 'evaluate', 'run'))
    if len(run_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@evaluate.run`.')
    
    evolve_functions = list(code_manipulation.yield_decorated(specification, 'equation', 'evolve'))
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@equation.evolve`.')
    
    return evolve_functions[0], run_functions[0]


def main(
    specification: str,
    inputs: Sequence[Any],
    config: config_lib.PTConfig,
    max_sample_nums: int | None,
    class_config: config_lib.ClassConfig,
    ood_inputs: Sequence[Any] = None,
    **kwargs
):
    """
    Launch a PT-LLMSR experiment.
    
    This is the parallel tempering variant of the LLMSR pipeline that uses
    multiple pools at different temperatures within each island to encourage
    diversity while maintaining high-quality solutions.
    
    Args:
        specification: The boilerplate code for the problem.
        inputs: The data instances for the problem.
        config: PT-enabled configuration.
        max_sample_nums: Maximum samples from LLM. 'None' means no limit.
        class_config: LLM and sandbox class configuration.
        ood_inputs: Optional OOD (out-of-distribution) data for diversity analysis.
        **kwargs: Additional arguments including 'log_dir'.
        
    Returns:
        Profiler object with experiment results.
    """
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)
    
    # Get log directory
    log_dir = kwargs.get('log_dir', None)
    
    # Create PT-enabled experience buffer
    database = PTExperienceBuffer(
        config.experience_buffer,
        template,
        function_to_evolve,
        log_dir=log_dir,
    )
    
    # Create profiler
    if log_dir is None:
        profiler = None
    else:
        profiler = profile.Profiler(log_dir)
    
    # Create evaluators with optional OOD inputs
    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(PTEvaluator(
            database,
            template,
            function_to_evolve,
            function_to_run,
            inputs,
            timeout_seconds=config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class,
            ood_inputs=ood_inputs,
        ))
    
    # Analyze initial template
    initial = template.get_function(function_to_evolve).body
    evaluators[0].analyse(initial, island_id=None, version_generated=None, pool_id=0, profiler=profiler)
    
    # Create samplers
    samplers = [
        PTSampler(
            database,
            evaluators,
            config.samples_per_prompt,
            max_sample_nums=max_sample_nums,
            llm_class=class_config.llm_class,
            config=config
        )
        for _ in range(config.num_samplers)
    ]
    
    # Run sampling loop
    for s in samplers:
        s.sample(profiler=profiler)
    
    # Export final swap logs
    if hasattr(database, '_export_swap_logs'):
        database._export_swap_logs()
    
    # Final PT metrics update and summary export
    if profiler:
        database.update_profiler_pt_metrics(profiler)
        if hasattr(profiler, 'export_tracking_summary'):
            profiler.export_tracking_summary()
    
    return profiler


def run_pt_llmsr(
    specification: str,
    inputs: Sequence[Any],
    max_sample_nums: int = 1000,
    pool_temperatures: Tuple[float, ...] = (10.0, 1.0),
    migration_size: int = 30,
    migration_interval: int = 50,
    population_size: int = 100,
    num_islands: int = 1,
    samples_per_prompt: int = 4,
    use_api: bool = True,
    api_model: str = "gpt-3.5-turbo",
    log_dir: str = "./pt_llmsr_logs",
    llm_class=None,
    sandbox_class=None,
):
    """
    Convenience function to run PT-LLMSR with common parameters.
    
    Args:
        specification: Problem specification code
        inputs: Input data for evaluation
        max_sample_nums: Maximum LLM samples
        pool_temperatures: Inverse temperatures for PT pools (high=cold, low=hot)
        migration_size: Number of swap attempts per migration
        migration_interval: Programs between migrations
        population_size: Max programs per pool
        num_islands: Number of islands (typically 1 for pure PT)
        samples_per_prompt: Samples per LLM prompt
        use_api: Whether to use LLM API
        api_model: API model name
        log_dir: Directory for logs
        llm_class: LLM class to use
        sandbox_class: Sandbox class for code execution
        
    Returns:
        Profiler with results
    """
    from llmsr.sampler import APILanguageModel
    from llmsr.evaluator import LocalSandbox
    
    if llm_class is None:
        llm_class = APILanguageModel
    if sandbox_class is None:
        sandbox_class = LocalSandbox
    
    # Create PT configuration
    pt_buffer_config = config_lib.PTExperienceBufferConfig(
        functions_per_prompt=2,
        num_islands=num_islands,
        population_size=population_size,
        pool_temperatures=pool_temperatures,
        migration_size=migration_size,
        migration_interval=migration_interval,
    )
    
    config = config_lib.PTConfig(
        experience_buffer=pt_buffer_config,
        samples_per_prompt=samples_per_prompt,
        use_api=use_api,
        api_model=api_model,
    )
    
    class_config = config_lib.ClassConfig(
        llm_class=llm_class,
        sandbox_class=sandbox_class,
    )
    
    return main(
        specification=specification,
        inputs=inputs,
        config=config,
        max_sample_nums=max_sample_nums,
        class_config=class_config,
        log_dir=log_dir,
    )


