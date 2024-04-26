# named_tuples.py
import numpy as np
from typing import NamedTuple, List, Dict

# Define the NamedTuple for updating the model
UpdateModelData = NamedTuple(
    'UpdateModelData',
    [
        ('state', np.ndarray),
        ('action', int),
        ('reward', float),
        ('next_state', np.ndarray),
        ('done', bool),
    ]
)