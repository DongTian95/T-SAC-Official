import copy
import os
import sys

from mprl.rl.critic import AbstractCritic
from mprl.util import util
from mprl.util.util_nanogpt_critic_old import *


class TSACOldNanogpt(AbstractCritic):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            **config,
    ):
        self.config = config
        self.dtype, self.device = util.parse_dtype_device(config["dtype"], config["device"])

        self.net1 = None
        self.net2 = None
        self.target_net1 = None
        self.target_net2 = None

        # structure specified parameters
        self.eta = config["update_rate"]  # used for Polyak averaging

        # build up gpt config
        self.gpt_config = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "n_layer": config["net_args"]["n_layer"],
            "n_head": config["net_args"]["n_head"],
            "n_embd": config["net_args"]["n_embd"],
            "block_size": config["net_args"]["block_size"],
            "dropout": config["net_args"]["dropout"],
            "bias": config["net_args"]["bias"],
            "dtype": config["dtype"],
            "device": config["device"],
        }
        self.gpt_config = GPTConfig(
            input_dim=state_dim,
            output_dim=action_dim,
            block_size=config["net_args"]["block_size"],
            n_layer=config["net_args"]["n_layer"],
            n_head=config["net_args"]["n_head"],
            n_embd=config["net_args"]["n_embd"],
            dropout=config["net_args"]["dropout"],
            gpt_name="tsac_critic_old_nanogpt",
        )

        self._create_network()

    def _create_network(self):
        # config1 = copy.deepcopy(self.gpt_config)
        # config1["name"] = self._critic_net_type + "_1"
        # config2 = copy.deepcopy(self.gpt_config)
        # config2["name"] = self._critic_net_type + "_2"

        self.net1 = GPT(self.gpt_config).to(device=self.device, dtype=self.dtype)
        self.net2 = GPT(self.gpt_config).to(device=self.device, dtype=self.dtype)
        self.net1.train()
        self.net2.train()

        self.target_net1 = copy.deepcopy(self.net1)
        self.target_net2 = copy.deepcopy(self.net2)
        self.target_net1.requires_grad_(False)
        self.target_net2.requires_grad_(False)
        # self.target_net1.eval()
        # self.target_net2.eval()

    def configure_optimizer(self, weight_decay, learning_rate, betas):
        """
        The optimizer is chosen to be AdamW
        @return: constructed optimizer
        """
        opt1 = self.net1.configure_optimizers(weight_decay=weight_decay,
                                             learning_rate=learning_rate,
                                             betas=betas,
                                             )
        opt2 = self.net2.configure_optimizers(weight_decay=weight_decay,
                                             learning_rate=learning_rate,
                                             betas=betas,
                                             )
        return opt1, opt2

    def generate_idx_tensors(self, batch_size: int, sequence_length: int, current_states_idx: int, step_length: int):
        # current_states_idx + step_length should be <= sequence_length
        if current_states_idx + step_length > sequence_length:
            step_length = sequence_length - current_states_idx

        # generate idx_c and idx_i tensors
        idx_c = util.to_ts(current_states_idx, self.dtype, self.device)
        idx_c = idx_c.view(1, -1).repeat(batch_size, 1)
        idx_a = torch.arange(current_states_idx, current_states_idx + step_length, dtype=self.dtype, device=self.device)
        idx_a = idx_a.view(1, -1).repeat(batch_size, 1)

        return idx_c, idx_a

    @staticmethod
    def critic(net, c_state, actions, idx_c, idx_a, no_absolute_idx=False):
        pred = net(state=c_state, action=actions)
        pred = pred[:, 1:, :]
        return pred

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()
