# update_model_data.py
from typing import NamedTuple
import numpy as np

class UpdateModelData(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

def get_update_model_data():
    # Implementation of get_update_model_data function
    pass