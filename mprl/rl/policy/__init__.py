from typing import Literal

from .abstract_policy import *
from .transformer_sac_policy import *
from .n_return_sac_policy import *
from .rlac_policy import *


def policy_factory(typ: Literal["TemporalCorrelatedPolicy", "BlackBoxPolicy"],
                   **kwargs):
    """
    Factory methods to instantiate a policy
    Args:
        typ: policy class type
        **kwargs: keyword arguments

    Returns:

    """
    return eval(typ + "(**kwargs)")
