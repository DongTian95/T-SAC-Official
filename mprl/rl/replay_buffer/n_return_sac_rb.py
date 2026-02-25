import os
import sys

import time
import logging
import numpy as np
import torch
import random
from collections import namedtuple, deque

import mprl.util as util

# Experience = namedtuple("Experience", ["states", "actions", "rewards", "dones",
#                                        "rewards_sum", "possibility", "masks"])
Experience = namedtuple("Experience", ["states", "actions", "rewards", "dones",
                                       "masks"])


class NReturnSACReplayBuffer:
    def __init__(self, buffer_size: int = 1e4, batch_size: int = None, dtype=torch.float32, device: str = "cpu",
                 **kwargs):
        self._batch_size = batch_size
        self.dtype, self.device = util.parse_dtype_device(dtype, device)

        # Initialize the replay buffer
        self._buffer = deque(maxlen=buffer_size)
        # self._probabilities = deque(maxlen=int(buffer_size))

        # set the rewards scaling for a better probability of prioritized sampling
        self._prb_reward_scaling = kwargs.get("prb_reward_scaling")

    def add(self, dataset_dict: dict):
        """
        Add an experience to the replay buffer
        Args:
            dataset_dict: dict
                state: shape: [num_env, num_times, dim_obs]
                action: shape: [num_env, num_times, num_dof * 2]
                reward: shape: [num_env, num_times]
                last_state: shape: [num_env, dim_obs]
                dones: [num_env, num_times]

        Returns: None
        """
        # Unpack the dataset
        states = dataset_dict["states"]
        actions = dataset_dict["actions"]
        rewards = dataset_dict["rewards"]
        dones = dataset_dict["dones"]
        masks = dataset_dict["masks"]

        # scaled_rewards = rewards * self._prb_reward_scaling
        # rewards_sum = torch.sum(scaled_rewards, dim=1)

        # By saving, all samples generated from different envs will be stored in different buffers
        for env_idx in range(states.shape[0]):
            experience = Experience(
                states=states[env_idx, ...].detach(),    # e.g. shape: [num_times, dim_obs]
                actions=actions[env_idx, ...].detach(),
                rewards=rewards[env_idx, ...].detach(),
                dones=dones[env_idx, ...].detach(),
                masks=masks[env_idx, ...].detach(),
                # rewards_sum=rewards_sum[env_idx, ...].item(),
                # possibility=None,    # Placeholder, will be updated later
            )
            self._buffer.append(experience)
            # self._probabilities.append(rewards_sum[env_idx, ...].item())

        # self.update_probabilities()

    def update_probabilities(self):
        if not self._probabilities:
            return

        max_prob = max(self._probabilities)
        exp_probabilities = np.exp([p - max_prob for p in self._probabilities])
        log_sum_exp = np.log(np.sum(exp_probabilities)) + max_prob  # log-sum-exp trick
        norm_probabilities = np.exp([p - log_sum_exp for p in self._probabilities])

        for idx, exp in enumerate(self._buffer):
            self._buffer[idx] = exp._replace(possibility=norm_probabilities[idx])

    def process_batch(self, batch):
        states, actions, rewards, last_states, dones, masks, segment_params_mean, segment_params_L = (
            [], [], [], [], [], [], [], [])

        for exp in batch:
            states.append(exp.states.to(self.device, self.dtype))
            actions.append(exp.actions.to(self.device, self.dtype))
            rewards.append(exp.rewards.to(self.device, self.dtype))
            dones.append(exp.dones.to(self.device, self.dtype))
            masks.append(exp.masks.to(self.device, self.dtype))

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        masks = torch.stack(masks)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "masks": masks,
        }

    def sample(self):
        assert self._batch_size <= len(self._buffer), "Batch size is larger than replay buffer"
        batch = random.sample(self._buffer, k=self._batch_size)
        return self.process_batch(batch)

    def prioritized_sample(self):
        assert self._batch_size <= len(self._buffer), "Batch size is larger than replay buffer"
        indices = np.random.choice(len(self._buffer), size=self._batch_size,
                                   p=[exp.possibility for exp in self._buffer], replace=False)
        batch = [self._buffer[i] for i in indices]
        return self.process_batch(batch)

    @property
    def batch_size(self):
        return self._batch_size

    def is_full(self):
        return len(self._buffer) >= self._buffer.maxlen

    def is_ready(self):
        return len(self._buffer) > 5 * self._batch_size
        # return len(self._buffer) >= self._batch_size

    def __len__(self):
        return len(self._buffer)


if __name__ == "__main__":  # Test function, DO NOT start the program from here in the working mode
    pass