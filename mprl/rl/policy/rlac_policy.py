import os
import sys

import torch
from sympy.physics.quantum.density import entropy

from mprl.util import util, MLP, TrainableVariable
from .abstract_policy import AbstractGaussianPolicy


class RLACPolicy(AbstractGaussianPolicy):
    def __init__(
            self,
            state_dim: int,
            dim_out: int,
            info_dim: int,  # no usage, to keep the compatibility to drp_step_based
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
        self.policy_step_length = kwargs.get("policy_step_length", 1)
        super(RLACPolicy, self).__init__(
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

    def _create_network(self):
        """
        Create policy net with given configuration

        Returns:
            None
        """

        # Two separate value heads: mean_val_net + cov_val_net
        self.mean_net = MLP(name=self._policy_net_type + "_mean",
                            dim_in=self.dim_in,
                            dim_out=self.dim_out * self.policy_step_length,
                            hidden_layers=util.mlp_arch_3_params(
                                **self.mean_net_args),
                            init_method=self.init_method,
                            out_layer_gain=self.out_layer_gain,
                            act_func_hidden=self.act_func_hidden,
                            act_func_last=self.act_func_last,
                            layer_norm=self.mean_layer_norm,
                            dtype=self.dtype,
                            device=self.device)

        # compute the output dimension of variance
        if self.std_only:
            # Only has diagonal elements
            dim_out_var = self.dim_out
        else:
            raise NotImplementedError

        if self.contextual_cov:
            if not self.feed_mean_to_variance:
                self.variance_net = MLP(name=self._policy_net_type + "_variance",
                                        dim_in=self.dim_in,
                                        dim_out=dim_out_var * self.policy_step_length,
                                        hidden_layers=util.mlp_arch_3_params(
                                            **self.variance_net_args),
                                        init_method=self.init_method,
                                        out_layer_gain=self.out_layer_gain,
                                        act_func_hidden=self.act_func_hidden,
                                        act_func_last=self.act_func_last,
                                        layer_norm=self.variance_layer_norm,
                                        dtype=self.dtype,
                                        device=self.device)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def policy(self, obs, provided_mean=None, action_range=1.0):
        """
        compute the mean cov of the action given state
        Args:
            obs: state
            provided_mean: is only valid when self.feed_mean_to_variance is True
            action_range: should be provided when self.feed_mean_to_variance is True and action_range is not 1.0
        """
        params_mean = self.mean_net(obs).view(-1, self.policy_step_length, self.dim_out)
        if self.contextual_cov:
            if self.contextual_cov and not self.feed_mean_to_variance:
                variance = self.variance_net(obs).view(-1, self.policy_step_length, self.dim_out)
                params_L = torch.stack(
                    [self._vector_to_cholesky(variance[:, i, :]) for i in range(self.policy_step_length)],
                    dim=1
                )
            else:
                if provided_mean is None:
                    params_mean_detach = params_mean
                else:
                    params_mean_detach = provided_mean
                params_mean_detach = torch.tanh(params_mean_detach) * action_range
                params_mean_detach = params_mean_detach.detach()

                params_L = self._vector_to_cholesky(self.variance_net(torch.cat([obs, params_mean_detach], dim=1)))
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

    def reset_network(self):
        """
        reset the network
        :return: None
        """
        self._create_network()

    def maha(self, params: torch.Tensor, params_other: torch.Tensor,
             params_L: torch.Tensor):
        """
        Compute the Mahalanobis distance of the policy

        Args:
            params:
            params_other:
            params_L: Cholesky matrix

        Returns:
            maha:  Mahalanobis distance of the policy
        """
        diff = (params - params_other)[..., None]

        # A new version of torch.triangular_solve(B, A).solution
        maha = torch.linalg.solve_triangular(params_L,
                                             diff,
                                             upper=False).pow(2).sum([-2, -1])
        return maha

    def precision(self, params_L: torch.Tensor):
        """
        Compute the precision of the policy

        Args:
            params_L: Cholesky matrix

        Returns:
            precision: precision of the policy
        """
        precision = torch.cholesky_solve(torch.eye(params_L.shape[-1],
                                                   dtype=params_L.dtype,
                                                   device=params_L.device),
                                         params_L, upper=False)
        return precision

    def eval(self):
        self.mean_net.eval()
        self.variance_net.eval()

    def train(self):
        self.mean_net.train()
        self.variance_net.train()