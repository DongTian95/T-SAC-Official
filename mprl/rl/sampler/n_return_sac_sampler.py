import os
import sys

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import mprl.util as util
from mprl.rl.critic import NReturnSACCritic
from mprl.rl.policy import NReturnSACPolicy
from mprl.rl.sampler import AbstractSampler


class NReturnSACSampler(AbstractSampler):
    def __init__(
            self,
            env_id: str,
            num_env_train: int = 1,
            num_env_test: int = 1,
            episodes_per_train_env: int = 1,
            episodes_per_test_env: int = 1,
            dtype: str = "torch.float32",
            device: str = "cpu",
            seed: int = 1,
            **kwargs,
    ):
        super().__init__()  # this line doesn't really do anything

        # Environment
        self.env_id = env_id

        self.num_env_train = num_env_train
        self.num_env_test = num_env_test
        self.episodes_per_train_env = episodes_per_train_env
        self.episodes_per_test_env = episodes_per_test_env

        self.dtype, self.device = util.parse_dtype_device(dtype=dtype, device=device)
        self.seed = seed

        # map env to cpu cores
        self.cpu_cores = kwargs.get("cpu_cores", None)

        # logging the task specified metrics
        self.task_specified_metrics = kwargs.get("task_specified_metrics", None)

        # render the test env
        self.render_test_env = kwargs.get("render_test_env", False)

        # get env disable info
        disable_train_env = kwargs.get("disable_train_env", False)
        disable_test_env = kwargs.get("disable_test_env", False)

        # get training and testing environments
        if not disable_train_env:
            self.train_envs = self.get_env(env_type="training")
        else:
            self.train_envs = None

        if not disable_test_env:
            self.test_envs = self.get_env(env_type="testing")
        else:
            self.test_envs = None

        # get one more environmen for debugging
        self.debug_env = self.get_env(env_type="debugging")

        # get settings parameters

        self.num_times = self.debug_env.envs[0].spec.max_episode_steps
        self.reward_scaling = kwargs.get("reward_scaling")
        self.step_length = kwargs.get("step_length")
        self.min_length = kwargs.get("min_length", 1)
        self.action_range = self.debug_env.envs[0].action_space.high
        if isinstance(self.action_range, np.ndarray):
            self.action_range = float(self.action_range[0])    # TODO: maybe not optimistic enough

        # padding settings to deal with trajectories variable lengths
        self.padding_on = kwargs.get("padding_on", False)

        # whether add timestamp
        self.add_timestamp=kwargs.get("add_timestamp", True)

    @staticmethod
    def make_envs(env_id: str, num_env: int, seed: int, render: bool, **kwargs):
        if render:
            assert num_env == 1, "Rendering only works with num_env=1"

        vec_env = SubprocVecEnv if num_env > 1 else DummyVecEnv     # Creates a simple vectorized wrapper for
        # multiple environments, calling each environment IN SEQUENCE on the current Python process

        def _make_env(env_id: str, seed: int, rank: int, render:bool, **kwargs):
            """
            get a function instance for creating an env
            """
            def _get_env():
                util.set_logger_level("ERROR")
                env = gym.make(id=env_id, render_mode="human" if render else None, **kwargs)
                env.reset(seed=seed + rank)
                return env

            return _get_env

        env_fns = [_make_env(env_id, seed=seed, rank=i, render=render, **kwargs) for i in range(num_env)]

        return vec_env(env_fns)

    def get_env(self, env_type: str = "training"):
        render = False

        if env_type == "training":
            num_env = self.num_env_train
            seed = self.seed
        elif env_type == "testing":
            num_env = self.num_env_test
            seed = self.seed + 10000
            render = self.render_test_env
        elif env_type == "debugging":
            num_env = 1
            seed = self.seed + 20000
        else:
            raise TypeError("Unknown env_type: {}".format(env_type))

        # make envs
        envs = self.make_envs(env_id=self.env_id, num_env=num_env, seed=seed, render=render)

        # map env to cpu cores to avoid cpu conflicts in HPC
        util.assign_env_to_cpu(num_env, envs, self.cpu_cores)

        return envs

    @torch.no_grad()
    def run(
            self,
            training: bool,
            policy: NReturnSACPolicy,
            critic: NReturnSACCritic,
            deterministic: bool = False,
            render: bool = False,
            task_specific_metrics: dict = None,
    ):
        if training:
            assert deterministic is False and render is False
            envs = self.train_envs
            episode_init_state = envs.reset()
            num_env = self.num_env_train
            ep_per_env = self.episodes_per_train_env
        else:
            envs = self.test_envs
            episode_init_state = envs.reset()
            num_env = self.num_env_test
            if render and num_env == 1:
                envs.render()
            ep_per_env = self.episodes_per_test_env

        # init env interactions steps
        num_total_env_steps = 0

        # init storage for rollout results
        list_step_states = list()       # intermediate storage, will not be returned
        list_step_actions = list()
        list_step_rewards = list()
        list_step_dones = list()

        # final storage, will be returned
        list_step_states_result = list()
        list_step_actions_result = list()
        list_step_rewards_result = list()
        list_step_dones_result = list()

        # storage task specified metrics
        if self.task_specified_metrics is not None:
            dict_task_specified_metrics = dict()
            for metric in self.task_specified_metrics:
                dict_task_specified_metrics[metric] = list()
        else:
            dict_task_specified_metrics = dict()

        # storage first states
        episode_init_state = util.to_ts(episode_init_state, dtype=self.dtype, device=self.device)
        list_step_states.append(episode_init_state.unsqueeze(1))    # [num_env, 1, dim_obs]

        for _ in range(ep_per_env):
            # init env indices
            env_indices = list(range(num_env))

            # main rollout loop
            while True:
                # policy prediction
                current_states_idx = len(list_step_states) - 1

                if self.add_timestamp == "only_action":
                    pass
                elif self.add_timestamp:
                    # add time stamp to episode_init_state
                    time_stamp = util.to_ts(current_states_idx, self.dtype, self.device)
                    time_stamp = time_stamp.repeat(num_env, 1)
                    episode_init_state = torch.cat((time_stamp, episode_init_state), dim=1)

                params_mean, params_L = policy.policy(episode_init_state)

                # sample actions
                actions = policy.sample(params_mean, params_L, use_mean=deterministic, require_grad=False)
                actions = torch.tanh(actions) * self.action_range   # follow the standard SAC implementation
                actions = util.to_np(actions)

                step_states, step_rewards, step_dones, step_infos = envs.step(actions)

                num_total_env_steps += len(env_indices)

                # apply rewards scaling
                if self.reward_scaling is not None and self.reward_scaling != 1.0:
                    step_rewards = step_rewards * self.reward_scaling

                # transform to tensor
                step_states = util.to_ts(step_states, dtype=self.dtype, device=self.device)
                step_actions = util.to_ts(actions, dtype=self.dtype, device=self.device)
                step_rewards = util.to_ts(step_rewards, dtype=self.dtype, device=self.device).view(num_env, -1)
                step_dones = util.to_ts(step_dones, dtype=self.dtype, device=self.device).view(num_env, -1)

                # save to storage
                list_step_states.append(step_states.unsqueeze(1))
                list_step_actions.append(step_actions.unsqueeze(1))
                list_step_rewards.append(step_rewards.unsqueeze(1))
                list_step_dones.append(step_dones.unsqueeze(1))

                # end of one episode
                if step_dones.any():
                    # get the dones trajectories
                    dones_indices = torch.nonzero(step_dones.view(-1), as_tuple=False).view(-1)
                    # if there's only one done environment, ensure dones_indices is a 1D tensor
                    if dones_indices.dim() == 0:
                        dones_indices = dones_indices.unsqueeze(0)

                    # process the data for the done env
                    for idx in dones_indices:
                        env_idx = idx.item()    # convert tensor to integer index
                        if env_idx not in env_indices:
                            continue
                        else:
                            env_indices.remove(env_idx)

                        # concat the data collected for this env
                        all_states = torch.cat(list_step_states, dim=1)
                        all_actions = torch.cat(list_step_actions, dim=1)
                        all_rewards = torch.cat(list_step_rewards, dim=1)
                        all_dones = torch.cat(list_step_dones, dim=1)

                        # access the data collected for this env
                        env_states = all_states[env_idx, ...]
                        env_actions = all_actions[env_idx, ...]
                        env_rewards = all_rewards[env_idx, ...]
                        env_dones = all_dones[env_idx, ...]

                        # save to the final results
                        list_step_states_result.append(env_states)
                        list_step_actions_result.append(env_actions)
                        list_step_rewards_result.append(env_rewards)
                        list_step_dones_result.append(env_dones)

                        # task specified metrics
                        if self.task_specified_metrics is not None:
                            for metric in self.task_specified_metrics:
                                metric_value = util.get_item_from_dicts(step_infos, metric)
                                metric_value = util.to_ts(metric_value, dtype=self.dtype, device=self.device)
                                dict_task_specified_metrics[metric].append(metric_value)

                    # for the next episode
                    if len(env_indices) == 0:
                        break

                # update episode init state
                episode_init_state = step_states

            # clear the buffers for this env after processing
            list_step_states = list()
            list_step_actions = list()
            list_step_rewards = list()
            list_step_dones = list()

            # reset the env
            episode_init_state = envs.reset()
            # save the first state
            episode_init_state = util.to_ts(episode_init_state, dtype=self.dtype, device=self.device)
            list_step_states.append(episode_init_state.unsqueeze(1))

        # init the final results
        results = dict()

        if self.padding_on:
            batch_size = len(list_step_states_result)     # this batch_size is actually the number of envs * ep_per_env
            max_sequence_length = self.num_times    # Desired sequence length after padding

            padded_states = []
            padded_actions = []
            padded_rewards = []
            padded_dones = []
            masks = []

            for i in range(batch_size):
                seq_states = list_step_states_result[i]
                seq_actions = list_step_actions_result[i]
                seq_rewards = list_step_rewards_result[i]
                seq_dones = list_step_dones_result[i]

                seq_length = seq_states.shape[0]

                # Create mask for the current sequence
                mask = torch.zeros(max_sequence_length, dtype=torch.bool, device=self.device)
                mask[:seq_length - 1] = True    # exclude the initial state
                masks.append(mask)

                # Helper function to pad a tensor
                def pad_tensor(tensor, pad_value=0, pad_length=max_sequence_length):
                    tensor_length, feature_length = tensor.size()
                    padded_tensor = torch.full(
                        (pad_length, feature_length),
                        pad_value,
                        dtype=tensor.dtype,
                        device=self.device,
                    )
                    padded_tensor[:tensor_length] = tensor
                    return padded_tensor

                # Pad each tensor in the sequence
                padded_states.append(pad_tensor(seq_states, pad_length=max_sequence_length + 1))    # init state
                padded_actions.append(pad_tensor(seq_actions))
                padded_rewards.append(pad_tensor(seq_rewards))
                padded_dones.append(pad_tensor(seq_dones, pad_value=1))     # 1 for done

            # stack the padded sequences and masks into batch tensors
            list_step_states_result = padded_states
            list_step_actions_result = padded_actions
            list_step_rewards_result = padded_rewards
            list_step_dones_result = padded_dones
            results["masks"] = torch.stack(masks, dim=0).unsqueeze(-1)
        else:   # no padding
            results["masks"] = torch.ones((len(list_step_states_result), self.num_times, 1),
                                          dtype=torch.bool, device=self.device)

        # form up return dictionary
        results["states"] = torch.stack(list_step_states_result, dim=0)
        results["actions"]= torch.stack(list_step_actions_result, dim=0)
        results["rewards"] = torch.stack(list_step_rewards_result, dim=0)
        results["dones"] = torch.stack(list_step_dones_result, dim=0)

        if self.task_specified_metrics is not None:
            for metric in self.task_specified_metrics:
                results[metric] = torch.cat(dict_task_specified_metrics[metric], dim=0)

        return results, num_total_env_steps
