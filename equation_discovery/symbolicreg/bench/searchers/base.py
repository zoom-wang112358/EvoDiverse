from typing import List, Optional
import numpy as np

from ..dataclasses import SEDTask, Equation


class BaseSearcher:
    def __init__(self, name) -> None:
        self._name = name

    def discover(self, task: SEDTask, ood_data: Optional[np.ndarray] = None) -> List[Equation]:
        '''
        Discover equations for the given task.
        
        Args:
            task: The search task with training data
            ood_data: Optional out-of-distribution test data for diversity analysis
        
        Return:
            equations
            aux
        '''
        raise NotImplementedError

    def __str__(self):
        return self._name