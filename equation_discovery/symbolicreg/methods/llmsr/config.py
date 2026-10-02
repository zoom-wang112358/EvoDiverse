"""Configuration for EvoDiverse equation search."""

from __future__ import annotations

import dataclasses

from typing import Type

import os

from llmsr import sampler

from llmsr import evaluator

@dataclasses.dataclass(frozen=True)
class PTExperienceBufferConfig:
    """Configures PT (Parallel Tempering) Experience Buffer parameters.
    
    Each island contains multiple parallel-tempering pools at different
    temperatures, enabling diversity through PT-style swaps.
    
    Args:
        functions_per_prompt (int): Number of previous hypotheses to include in prompts
        num_islands (int): Number of islands (set to 1 for single-island PT mode)
        population_size (int): Maximum programs per pool
        pool_temperatures (tuple): Inverse temperatures (beta) for each pool.
            Higher beta = colder = more exploitation.
            Default: (0.8, 0.01) for cold and hot pools.
        migration_size (int): Number of swap attempts per migration
        migration_interval (int): Number of programs registered between migrations
        reset_period (int): Seconds between weakest island resets
        cluster_sampling_temperature_init (float): Initial softmax temperature for sampling
        cluster_sampling_temperature_period (int): Period for temperature decay
        pool_selection_strategy (str): How to select pools for prompts:
            - "uniform": equal probability for all pools
            - "cold_biased": favor cold pool for exploitation
        log_every_migrations (int): How often to export swap logs
        
        Adaptive Boltzmann constant (xi) parameters for swap rate control:
        xi_init (float): Initial Boltzmann constant for computing PT acceptance.
            This scales the temperature difference in the acceptance probability.
        swap_rate_target (float): Target swap rate (τ). Default: 0.3
        swap_rate_tolerance (float): Tolerance (ε) around the target. Adaptation
            occurs outside [τ - ε/2, τ + ε/2]. Default: 0.1
        swap_rate_window_size (int): Window size (L) for computing rolling swap rate.
            Adaptation happens after each complete window. Default: 5
        xi_decay (float): Multiplicative factor to reduce xi when swap rate too high.
            Applied when mean(Γ) >= τ + ε/2. Default: 0.8
        xi_growth (float): Multiplicative factor to increase xi when swap rate too low.
            Applied when mean(Γ) <= τ - ε/2. Default: 1.2
        xi_min (float): Minimum allowed value for xi. Default: 0.1
        xi_max (float): Maximum allowed value for xi. Default: 50.0
    """
    functions_per_prompt: int = 2
    num_islands: int = 1  # Single island with PT pools inside
    population_size: int = 1000
    pool_temperatures: tuple = (0.8, 0.01)  # (cold/hot beta)
    migration_size: int = 10
    migration_interval: int = 5  # Perform migration every N programs
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000
    pool_selection_strategy: str = "uniform"
    log_every_migrations: int = 1
    
    # Adaptive Boltzmann constant (xi) parameters for swap rate control
    xi_init: float = 2.5  # Initial Boltzmann constant
    swap_rate_target: float = 0.3  # Target swap rate (τ)
    swap_rate_tolerance: float = 0.1  # Tolerance (ε)
    swap_rate_window_size: int = 5  # Window size (L) for rolling swap rate
    xi_decay: float = 0.8 # Decay factor when swap rate too high
    xi_growth: float = 1.2  # Growth factor when swap rate too low
    xi_min: float = 0.1  # Minimum xi value
    xi_max: float = 50.0  # Maximum xi value
    
    # Buffer snapshot saving for diversity analysis
    snapshot_interval: int = 50  # Save buffer every N iterations (0 to disable)

@dataclasses.dataclass(frozen=True)
class PTConfig:
    """Configuration for EvoDiverse equation search and program evaluation.
    
    Args:
        experience_buffer: PT-enabled experience buffer configuration
        num_samplers (int): Number of parallel samplers
        num_evaluators (int): Number of parallel evaluators
        samples_per_prompt (int): Number of hypotheses per prompt
        evaluate_timeout_seconds (int): Hypothesis evaluation timeout
        use_api (bool): API usage flag
        api_model (str): Model name for API calls
    """
    experience_buffer: PTExperienceBufferConfig = dataclasses.field(default_factory=PTExperienceBufferConfig)
    num_samplers: int = 1
    num_evaluators: int = 1
    samples_per_prompt: int = 4
    evaluate_timeout_seconds: int = 30
    use_api: bool = False
    api_model: str = "gpt-3.5-turbo"

@dataclasses.dataclass()
class ClassConfig:
    llm_class: Type[sampler.LLM]
    sandbox_class: Type[evaluator.Sandbox]
