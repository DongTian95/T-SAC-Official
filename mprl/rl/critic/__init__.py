from typing import Literal

from .abstract_critic import *
from .transformer_sac_critic import *
from .tsac_old_nanogpt import *
from .n_return_sac_critic import *
from .original_sac_critic import *



def critic_factory(typ: Literal["ValueFunction"],
                   **kwargs):
    """
    Factory methods to instantiate a critic
    Args:
        typ: critic class type
        **kwargs: keyword arguments

    Returns:

    """
    return eval(typ + "(**kwargs)")
