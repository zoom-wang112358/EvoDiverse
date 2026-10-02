

"""
Sampler for PT-LLMSR that works with the PT Experience Buffer.

Extends the base sampler to track pool IDs and route samples appropriately.
"""
from __future__ import annotations

from typing import Collection, Sequence, Type
import numpy as np
import time
import asyncio

from llmsr.pt_evaluator import PTEvaluator
from llmsr.pt_buffer import PTExperienceBuffer
from llmsr import config as config_lib
from llmsr import profile
from llmsr.sampler import LLM, APILanguageModel


class PTSampler:
    """
    Sampler for PT-LLMSR that generates programs and routes them to pools.
    
    Similar to the base sampler but tracks which pool generated the prompt
    and ensures new programs are evaluated and registered in the correct pool.
    """
    
    _global_samples_nums: int = 1

    def __init__(
        self,
        database: PTExperienceBuffer,
        evaluators: Sequence[PTEvaluator],
        samples_per_prompt: int,
        config: config_lib.PTConfig,
        max_sample_nums: int | None = None,
        llm_class: Type[LLM] = APILanguageModel,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config

    def sample(self, **kwargs):
        """
        Continuously get prompts, sample programs, and send them for analysis.
        
        This tracks the pool_id from prompts and routes samples back to the
        appropriate pool for PT-style evolution.
        """
        while True:
            # Check stopping condition
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                print("Reached max sample nums, stopping.")
                break

            profiler: profile.Profiler = kwargs.get('profiler', None)
            if profiler:
                if profiler._cur_best_program_score == 0:
                    print("Perfect score achieved, stopping.")
                    break

            # Get prompt (includes pool_id for routing)
            prompt = self._database.get_prompt()
            
            reset_time = time.time()

            num_concurrent = 20
            queue = asyncio.Queue(maxsize=self._samples_per_prompt)
            schedule_queue = asyncio.Queue(maxsize=num_concurrent)

            async def sample_program():
                await schedule_queue.put(1)
                completion = await self._llm.async_draw_single_sample(prompt.code, self.config)
                await schedule_queue.get()
                await queue.put(completion)

            async def process_sampled_program():
                for count in range(self._samples_per_prompt):
                    sample = await queue.get()
                    sample_time = (time.time() - reset_time) / (count + 1)

                    self._global_sample_nums_plus_one()
                    cur_global_sample_nums = self._get_global_sample_nums()
                    
                    chosen_evaluator: PTEvaluator = np.random.choice(self._evaluators)
                    chosen_evaluator.analyse(
                        sample,
                        prompt.island_id,
                        prompt.version_generated,
                        pool_id=prompt.pool_id,  # Route to the source pool
                        **kwargs,
                        global_sample_nums=cur_global_sample_nums,
                        sample_time=sample_time
                    )

            loop = asyncio.new_event_loop()
            tasks = [
                loop.create_task(sample_program()) for _ in range(self._samples_per_prompt)
            ] + [loop.create_task(process_sampled_program())]
            
            try:
                loop.run_until_complete(asyncio.gather(*tasks))
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                loop.close()
            
            # Update PT metrics in profiler periodically (every 10 samples)
            cur_global_sample_nums = self._get_global_sample_nums()
            if profiler and cur_global_sample_nums % 10 == 0:
                self._database.update_profiler_pt_metrics(profiler)

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1


