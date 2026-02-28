import copy
import os
import sys

from mprl.rl.critic import AbstractCritic
from mprl.util.util_nanogpt import *


class TransformerSACCritic(AbstractCritic):
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
            "use_layer_norm": config["net_args"].get("use_layer_norm", False),
        }

        self._create_network()

    def _create_network(self):
        config1 = copy.deepcopy(self.gpt_config)
        config1["name"] = self._critic_net_type + "_1"
        config2 = copy.deepcopy(self.gpt_config)
        config2["name"] = self._critic_net_type + "_2"

        self.net1 = TrajectoryQfunctionGPT(**config1)
        self.net2 = TrajectoryQfunctionGPT(**config2)
        self.net1.train()
        self.net2.train()

        self.target_net1 = copy.deepcopy(self.net1)
        self.target_net2 = copy.deepcopy(self.net2)
        self.target_net1.requires_grad_(False)
        self.target_net2.requires_grad_(False)
        self.target_net1.eval()
        self.target_net2.eval()

    def configure_optimizer(self, weight_decay, learning_rate, betas):
        """
        The optimizer is chosen to be AdamW
        @return: constructed optimizer
        """
        opt1 = self.net1.configure_optimizer(weight_decay=weight_decay,
                                             learning_rate=learning_rate,
                                             betas=betas,
                                             device_type=self.config["device"])
        opt2 = self.net2.configure_optimizer(weight_decay=weight_decay,
                                             learning_rate=learning_rate,
                                             betas=betas,
                                             device_type=self.config["device"])
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

    def save_weights(self, log_dir: str, epoch: int):
        """
        Save NN weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """
        self.net1.save(log_dir, epoch)
        self.net2.save(log_dir, epoch)
        
    def load_weights(self, log_dir: str, epoch: int):
        """
        Load NN weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
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
    def critic(net, c_state, actions, idx_c, idx_a, no_absolute_idx=False):
        return net(c_state=c_state, actions=actions, idx_c=idx_c, idx_a=idx_a, no_absolute_idx=no_absolute_idx)

    def eval(self):
        self.net1.eval()
        self.net2.eval()

    def train(self):
        self.net1.train()
        self.net2.train()

    def reset_network(self):
        self._create_network()

    def parameters(self) -> list:
        return [self.net1.parameters() + self.net2.parameters()]