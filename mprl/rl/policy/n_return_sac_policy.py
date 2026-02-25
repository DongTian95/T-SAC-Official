import os
import sys

import torch
import multiprocessing

import mprl.util as util
from .abstract_policy import AbstractGaussianPolicy


class NReturnSACPolicy(AbstractGaussianPolicy):
    def __init__(
            self,
            state_dim: int,
            dim_out: int,
            mean_net_args: dict,
            variance_net_args: dict,
            init_method: str,
            out_layer_gain: float,
            act_func_hidden: str,
            act_func_last: str,
            dtype: str = "torch.float32",
            device: str = "cpu",
            **kwargs,
    ):
        super(NReturnSACPolicy, self).__init__(
            dim_in=state_dim,
            dim_out=dim_out,
            mean_net_args=mean_net_args,
            variance_net_args=variance_net_args,
            init_method=init_method,
            out_layer_gain=out_layer_gain,
            act_func_hidden=act_func_hidden,
            act_func_last=act_func_last,
            dtype=dtype,
            device=device,
            **kwargs,
        )
        
    def policy(self, obs):
        """
        compute the mean cov of the action given state
        Args:
            obs: observation [*add_dim, dim_obs]
        Returns:
            mean and cholesky of the action
        """
        params_mean = self.mean_net(obs)
        if self.contextual_cov:
            params_L = self._vector_to_cholesky(self.variance_net(obs))
        else:
            L_vector = util.add_expand_dim(self.variance_net.variable, [0],
                                           [obs.shape[0]])
            params_L = self._vector_to_cholesky(L_vector)
        return params_mean, params_L

    def sample(self, params_mean, params_L, use_mean=False, require_grad=True):
        """
        rsample an action
        """
        if not use_mean:
            # sample trajectory
            mvn = torch.distributions.MultivariateNormal(
                loc=params_mean, scale_tril=params_L, validate_args=False
            )
            smp_params = mvn.rsample()
        else:
            smp_params = params_mean

        # remove gradient if not required
        if not require_grad:
            smp_params = smp_params.detach()

        return smp_params

    def log_prob(self, smp_params, params_mean, params_L, **kwargs):
        """
        compute the log probability of the sampled action
        """
        # form up trajectory distribution
        mvn = torch.distributions.MultivariateNormal(
            loc=params_mean,
            scale_tril=params_L,
            validate_args=False,
        )
        # compute log probability
        log_prob = mvn.log_prob(smp_params)

        return log_prob

    def entropy(self, params: [torch.Tensor, torch.Tensor]):
        """
        compute the entropy of the policy
        """
        # split mean and cholesky
        params_mean, params_L = params

        # for up a distribution
        mvn = torch.distributions.MultivariateNormal(
            loc=params_mean,
            scale_tril=params_L,
            validate_args=False,
        )
        entropy = mvn.entropy()
        return entropy

    def covariance(self, params_L: torch.Tensor):
        """
        compute the covariance of the policy
        """
        params_cov = torch.einsum('...ij,...kj->...ik', params_L, params_L)
        return params_cov

    def log_determinant(self, params_L: torch.Tensor):
        """
        compute the log_determinant of the policy
        """
        log_det = 2 * params_L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        return log_det
