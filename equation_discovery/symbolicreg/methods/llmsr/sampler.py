""" Class for sampling new program skeletons. """

from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type

import numpy as np

import time

from llmsr import config as config_lib





import asyncio

from openai import AsyncOpenAI


class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

def _extract_body(sample: str, config: config_lib.PTConfig) -> str:
    """
    Extract the function body from a response sample, removing any preceding descriptions
    and the function signature. Preserves indentation.
    ------------------------------------------------------------------------------------------------------------------
    Input example:
    ```
    This is a description...
    def function_name(...):
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    Output example:
    ```
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    If no function definition is found, returns the original sample.
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    
    for lineno, line in enumerate(lines):
        # find the first 'def' program statement in the response
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    
    if find_def_declaration:
        # Preserve indentation supplied by API completions.
        if config.use_api:
            code = ''
            for line in lines[func_body_lineno + 1:]:
                code += line + '\n'
        
        # Indent function bodies returned without a leading indent.
        else:
            code = ''
            indent = '    '
            for line in lines[func_body_lineno + 1:]:
                if line[:4] != indent:
                    line = indent + line
                code += line + '\n'
        
        return code
    
    return sample


class APILanguageModel(LLM):
    """Language-model adapter for OpenAI-compatible chat and Responses APIs.

    A client lives for one request so successive search batches can safely use
    separate event loops. SDK retries are bounded and failures reach the caller.
    """
    def __init__(self, samples_per_prompt, api_url=None, api_key=None, trim=True):
        super().__init__(samples_per_prompt)
        self._api_url = api_url
        self._api_key = api_key
        self._trim = trim
        self._instruction_prompt = (
            'You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. '
            '                             Complete the \'equation\' function below, considering the physical meaning and relationships of inputs.\n\n')

    @staticmethod
    def _use_responses_api(model_name):
        name = (model_name or '').lower()
        return name.startswith('gpt-5') and not name.startswith('gpt-5-chat')

    def draw_samples(self, prompt, config):
        async def batch():
            return await asyncio.gather(*(
                self.async_draw_single_sample(prompt, config)
                for _ in range(self._samples_per_prompt)))
        return asyncio.run(batch())

    async def async_draw_single_sample(self, prompt, config):
        if not config.use_api:
            raise ValueError('Use an OpenAI-compatible endpoint for local or hosted models.')
        messages = [{'role': 'user', 'content': '\n'.join([self._instruction_prompt, prompt])}]
        async with AsyncOpenAI(base_url=self._api_url, api_key=self._api_key,
                               timeout=60.0, max_retries=2) as client:
            if self._use_responses_api(config.api_model):
                response = await client.responses.create(
                    model=config.api_model, input=messages,
                    reasoning={'effort': getattr(config, 'reasoning_effort', 'medium')})
                text = response.output_text
            else:
                response = await client.chat.completions.create(
                    model=config.api_model, messages=messages)
                text = response.choices[0].message.content
        if not text:
            raise RuntimeError('The model returned an empty completion.')
        return _extract_body(text, config) if self._trim else text
