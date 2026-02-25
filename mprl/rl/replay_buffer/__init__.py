from typing import Literal

from .transformer_sac_replay_buffer import *
from .n_return_sac_rb import *


def replay_buffer_factory(typ: Literal["DRPReplayBuffer", "ReplayBuffer"],
                          **kwargs):
    """
    Factory methods to instantiate a replay buffer
    Args:
        typ: replay buffer class type
        **kwargs: keyword arguments

    Returns:

    """
    return eval(typ + "(**kwargs)")