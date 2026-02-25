import os
import sys

import torch
import copy

from mprl.rl.critic import AbstractCritic
import mprl.util as util
from mprl.util import MLP


class OriginalSACCritic(AbstractCritic):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            **config,
    ):
        self.config = config
        self.hidden = config["hidden"]

        self.dtype, self.device = util.parse_dtype_device(config["dtype"], config["device"])

        self.init_method = config["init_method"]
        self.out_layer_gain = config["out_layer_gain"]
        self.act_func_hidden = config["act_func_hidden"]
        self.act_func_last = config["act_func_last"]

        self.net1 = None
        self.net2 = None
        self.target_net1 = None
        self.target_net2 = None

        # structure specified parameters
        self.eta = config["update_rate"]    # used for Polyak averaging

        # get the input and output dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim

        # create network
        self._create_network()

    def _create_network(self):
        self.net1 = MLP(
            name=self._critic_net_type + "_1",
            dim_in=self.state_dim + self.action_dim,
            dim_out=1,
            hidden_layers=util.mlp_arch_3_params(**self.hidden),
            init_method=self.init_method,
            out_layer_gain=self.out_layer_gain,
            act_func_hidden=self.act_func_hidden,
            act_func_last=self.act_func_last,
            dtype=self.dtype,
            device=self.device,
        )
        self.net2 = MLP(
            name=self._critic_net_type + "_2",
            dim_in=self.state_dim + self.action_dim,
            dim_out=1,
            hidden_layers=util.mlp_arch_3_params(**self.hidden),
            init_method=self.init_method,
            out_layer_gain=self.out_layer_gain,
            act_func_hidden=self.act_func_hidden,
            act_func_last=self.act_func_last,
            dtype=self.dtype,
            device=self.device,
        )
        self.net1.train()
        self.net2.train()

        self.target_net1 = copy.deepcopy(self.net1)
        self.target_net2 = copy.deepcopy(self.net2)
        self.target_net1.requires_grad_(False)
        self.target_net2.requires_grad_(False)
        self.target_net1.eval()
        self.target_net2.eval()

        # save the critic network parameters
        self.critic1_net_params = self.net1.parameters()
        self.critic2_net_params = self.net2.parameters()

    def configure_optimizer(self, weight_decay, learning_rate, betas):
        """
        The optimizer is chosen to be AdamW
        """
        critic_opt1 = torch.optim.Adam(
            params=self.critic1_net_params,
            weight_decay=weight_decay,
            lr=learning_rate,
            betas=betas,
        )
        
        critic_opt2 = torch.optim.Adam(
            params=self.critic2_net_params,
            weight_decay=weight_decay,
            lr=learning_rate,
            betas=betas,
        )
        
        return critic_opt1, critic_opt2
    
    def save_weights(
            self,
            log_dir: str,
            epoch: int,
    ):
        self.net1.save(log_dir, epoch)
        self.net2.save(log_dir, epoch)

    def load_weights(self, log_dir: str, epoch: int):
        self.net1.load(log_dir, epoch)
        self.net2.load(log_dir, epoch)
        self.net1.train()
        self.net2.train()

        self.target_net1 = copy.deepcopy(self.net1)
        self.target_net2 = copy.deepcopy(self.net2)
        self.target_net1.requires_grad_(False)
        self.target_net2.requires_grad_(False)
        self.target_net1.eval()
        self.target_net2.eval()

    @staticmethod
    def critic(net, c_state, action):
        return net(torch.cat([c_state, action], dim=-1))

    def eval(self):
        self.net1.eval()
        self.net2.eval()

    def train(self):
        self.net1.train()
        self.net2.train()
