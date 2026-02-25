from typing import Literal

from .abstract_agent import *
from .transformer_sac import *
from .transformer_sac_multiprocessing import *
from .n_return_sac_agent import *
from .rlac import *
from .rlac_multiprocessing import *


def agent_factory(typ: Literal["TemporalCorrelatedAgent"],
                  **kwargs):
    """
    Factory methods to instantiate an agent
    Args:
        typ: agent class type
        **kwargs: keyword arguments

    Returns:

    """
    return eval(typ + "(**kwargs)")
