import io
import os
import sys

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import contextlib

import mprl.util as util
from mprl.rl.critic import TransformerSACCritic
from mprl.rl.policy import RLACPolicy
from mprl.rl.sampler import AbstractSampler


class RLACSampler(AbstractSampler):
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
            self.train_env_init_state = self.train_envs.reset()
            self.reset_at_beginning = kwargs.get("reset_at_beginning", False)
            self.current_step = 0
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
        self.min_length = kwargs.get("min_length", 1)
        self.step_length = kwargs.get("step_length")
        self.max_length = kwargs.get("max_length", self.num_times)
        self.action_range = self.debug_env.envs[0].action_space.high
        if isinstance(self.action_range, np.ndarray):
            self.action_range = float(self.action_range[0])    # TODO: maybe not optimistic enough

        # padding settings to deal with trajectories variable lengths
        self.padding_on = kwargs.get("padding_on", False)

        # whether add timestamp
        self.add_timestamp=kwargs.get("add_timestamp", False)

        # whether use done for target
        self.use_done_for_target=kwargs.get("use_done_for_target", True)

        # whether cut the whole trajectory into segments
        self.cut_segment = kwargs.get("cut_segment", 1)

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
            policy: RLACPolicy,
            critic: TransformerSACCritic,
            deterministic: bool = False,
            render: bool = False,
            task_specific_metrics: dict = None,
    ):
        if training:
            return self.run_train(training, policy, critic, deterministic, render, task_specific_metrics)
        else:
            return self.run_test(training, policy, critic, deterministic, render, task_specific_metrics)

    @torch.no_grad()
    def run_train(
            self,
            training: bool,
            policy: RLACPolicy,
            critic: TransformerSACCritic,
            deterministic: bool = False,
            render: bool = False,
            task_specific_metrics: dict = None,
    ):
        assert deterministic is False and render is False
        envs = self.train_envs
        if self.reset_at_beginning and self.current_step >= self.num_times:
            episode_init_state = envs.reset()
            self.current_step = 0
        else:
            episode_init_state = self.train_env_init_state
        num_env = self.num_env_train
        ep_per_env = self.episodes_per_train_env

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

        # main rollout loop
        for ep_idx in range(ep_per_env):
            while True:
                # policy prediction
                current_states_idx = len(list_step_states) - 1

                if self.add_timestamp and self.add_timestamp != "only_action":
                    # add time stamp to episode_init_state
                    time_stamp = util.to_ts(current_states_idx, self.dtype, self.device)
                    time_stamp = time_stamp.repeat(num_env, 1)
                    episode_init_state = torch.cat((time_stamp, episode_init_state), dim=1)

                params_mean, params_L = policy.policy(episode_init_state)

                # sample actions
                actions = policy.sample(params_mean, params_L, use_mean=deterministic, require_grad=False).view(num_env, -1)
                actions = torch.tanh(actions) * self.action_range   # follow the standard SAC implementation
                actions = util.to_np(actions)

                step_states, step_rewards, step_dones, step_infos = envs.step(actions)

                num_total_env_steps += num_env

                # apply rewards scaling
                if self.reward_scaling is not None and self.reward_scaling != 1.0:
                    step_rewards = step_rewards * self.reward_scaling

                # get the truncation state from info
                truncated = np.array(
                    [info.get("TimeLimit.truncated", False) for info in step_infos],
                    dtype=bool,
                )
                if not self.use_done_for_target and training:
                    step_dones = np.logical_and(step_dones.astype(bool), np.logical_not(truncated))

                # transform to tensor
                step_states = util.to_ts(step_states, dtype=self.dtype, device=self.device)
                step_actions = util.to_ts(actions, dtype=self.dtype, device=self.device)
                step_rewards = util.to_ts(step_rewards, dtype=self.dtype, device=self.device).view(num_env, -1)
                step_dones = util.to_ts(step_dones, dtype=self.dtype, device=self.device).view(num_env, -1)

                if truncated.any() and not self.use_done_for_target:
                    # Build a homogeneous [num_env, ...] tensor
                    truncation_states = []
                    for idx, info in enumerate(step_infos):
                        term = info.get("terminal_observation")
                        if term is not None:
                            obs = util.to_ts(term, dtype=self.dtype, device=self.device)
                        else:
                            obs = step_states[idx, ...]  # already correct dtype/device
                        truncation_states.append(obs)

                    # stack into [num_env, ...]
                    truncation_states = torch.stack(truncation_states, dim=0)

                    # truncation state must be at the end of the small episode
                    if len(list_step_states) != self.max_length:
                        # keep shape; mark rewards invalid
                        if isinstance(step_rewards, np.ndarray):
                            step_rewards = np.full_like(step_rewards, np.nan, dtype=np.float32)
                        elif torch.is_tensor(step_rewards):
                            step_rewards = torch.full_like(step_rewards, float('nan'))
                        else:
                            step_rewards = float('nan')

                # save to storage
                if truncated.any() and not self.use_done_for_target:
                    list_step_states.append(truncation_states.view(num_env, 1, -1))
                else:
                    list_step_states.append(step_states.unsqueeze(1))
                list_step_actions.append(step_actions.unsqueeze(1))
                list_step_rewards.append(step_rewards.unsqueeze(1))
                list_step_dones.append(step_dones.unsqueeze(1))

                # end of one episode
                if step_dones.any():
                    pass    # here should be the dealing of policy chunking

                if len(list_step_states) > self.max_length:
                    # concatenate all steps
                    list_step_states = torch.cat(list_step_states, dim=1)
                    list_step_actions = torch.cat(list_step_actions,dim=1)
                    list_step_rewards = torch.cat(list_step_rewards, dim=1)
                    list_step_dones = torch.cat(list_step_dones, dim=1)

                    # save to the final results
                    list_step_states_result.append(list_step_states)
                    list_step_actions_result.append(list_step_actions)
                    list_step_rewards_result.append(list_step_rewards)
                    list_step_dones_result.append(list_step_dones)

                    # reset storage
                    list_step_states = list()
                    list_step_actions = list()
                    list_step_rewards = list()
                    list_step_dones = list()

                    # task specified metrics
                    if self.task_specified_metrics is not None:
                        for metric in self.task_specified_metrics:
                            metric_value = util.get_item_from_dicts(step_infos, metric)
                            metric_value = util.to_ts(metric_value, self.dtype, self.device)
                            dict_task_specified_metrics[metric].append(metric_value)

                    # save the initial state of the next episode
                    episode_init_state = step_states
                    list_step_states.append(episode_init_state.unsqueeze(1))
                    self.current_step += self.max_length

                    # for the next episode
                    break

                # update episode init state
                episode_init_state = step_states

        self.train_env_init_state = episode_init_state

        # form up return dictionary
        results = dict()
        results["states"] = torch.cat(list_step_states_result, dim=0)
        results["actions"]= torch.cat(list_step_actions_result, dim=0)
        results["rewards"] = torch.cat(list_step_rewards_result, dim=0)
        results["dones"] = torch.cat(list_step_dones_result, dim=0)
        results["masks"] = torch.ones((num_env * ep_per_env, self.max_length, 1), dtype=torch.bool, device=self.device)

        if self.task_specified_metrics is not None:
            for metric in self.task_specified_metrics:
                results[metric] = torch.cat(dict_task_specified_metrics[metric], dim=0)

        return results, num_total_env_steps

    @torch.no_grad()
    def run_test(
            self,
            training: bool,
            policy: RLACPolicy,
            critic: TransformerSACCritic,
            deterministic: bool = False,
            render: bool = False,
            task_specific_metrics: dict = None,
    ):
        assert training is False
        envs = self.test_envs
        episode_init_state = envs.reset()
        num_env = self.num_env_test
        if render and num_env == 1:
            envs.render()
        ep_per_env = self.episodes_per_test_env

        # # init env interactions steps
        # num_total_env_steps = 0

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

        # get the Batch, policy_step_length and action_size
        B = num_env
        T = policy.policy_step_length
        F = self.debug_env.envs[0].action_space.shape[0]

        # storage first states
        episode_init_state = util.to_ts(episode_init_state, dtype=self.dtype, device=self.device)
        list_step_states.append(episode_init_state.unsqueeze(1))    # [num_env, 1, dim_obs]
        actions_tbe = util.to_np(torch.zeros((B, T, F), dtype=self.dtype, device=self.device))
        actions_mask = torch.zeros((B, T), dtype=self.dtype, device=self.device)

        # init storage for done dealing
        terminal_record = list()  # <- should contain tuple (time_step, env_idx, terminal_state)

        for idx_per_env in range(ep_per_env):
            # init env indices, it will decide how many envs are active
            env_indices = list(range(num_env))

            # main rollout loop
            while True:
                # policy prediction
                current_states_idx = len(list_step_states) - 1

                if self.add_timestamp and self.add_timestamp != "only_action":
                    # add time stamp to episode_init_state
                    time_stamp = util.to_ts(current_states_idx, self.dtype, self.device)
                    time_stamp = time_stamp.repeat(num_env, 1)
                    episode_init_state = torch.cat((time_stamp, episode_init_state), dim=1)

                if (actions_mask[:, 0] == 0).any():   # generate new actions if fake actions exist
                    # get the fake actions idx
                    fake_actions_idx = util.to_np((actions_mask[:, 0] == 0).nonzero(as_tuple=True)[0])
                    params_mean, params_L = policy.policy(episode_init_state[fake_actions_idx, ...],
                                                          action_range=self.action_range)

                    # sample actions
                    new_actions = policy.sample(params_mean, params_L, use_mean=deterministic, require_grad=False)
                    new_actions = torch.tanh(new_actions) * self.action_range   # follow the standard SAC implementation
                    actions_tbe[fake_actions_idx, ...] = util.to_np(new_actions)
                    actions_mask[fake_actions_idx, ...] = 1

                # get actions and delete the executed actions
                actions = actions_tbe[:, 0, :]
                actions_tbe = actions_tbe[:, 1:, :]
                actions_mask = actions_mask[:, 1:]

                # pad the actions_tbe(np.array) and actions_mask(tensor) to the length T
                actions_tbe = np.pad(actions_tbe, ((0, 0), (0, T - actions_tbe.shape[1]), (0, 0)), mode='constant')
                actions_mask = torch.cat((actions_mask, torch.zeros((B, T - actions_mask.shape[1]), dtype=self.dtype, device=self.device)), dim=1)

                step_states, step_rewards, step_dones, step_infos = envs.step(actions)
                truncated = np.array(
                    [info.get("TimeLimit.truncated", False) for info in step_infos],
                    dtype=bool,
                )
                if not self.use_done_for_target and training:
                    # SB3 VecEnv: list/tuple of dicts (Dummy/Subproc/SubVector)
                    # keep only true terminals (terminated=True, not time-limit)
                    step_dones = np.logical_and(step_dones.astype(bool), np.logical_not(truncated))
                # num_total_env_steps += len(env_indices)

                # apply rewards scaling
                if self.reward_scaling is not None and self.reward_scaling != 1.0:
                    step_rewards = step_rewards * self.reward_scaling

                # transform to tensor
                step_states = util.to_ts(step_states, dtype=self.dtype, device=self.device)
                step_actions = util.to_ts(actions, dtype=self.dtype, device=self.device)
                step_rewards = util.to_ts(step_rewards, dtype=self.dtype, device=self.device).view(num_env, -1)
                step_dones = util.to_ts(step_dones, dtype=self.dtype, device=self.device).view(num_env, -1)
                truncated = util.to_ts(truncated, dtype=torch.bool, device=self.device).view(num_env, -1)

                # save to storage
                list_step_states.append(step_states.unsqueeze(1))
                list_step_actions.append(step_actions.unsqueeze(1))
                list_step_rewards.append(step_rewards.unsqueeze(1))
                list_step_dones.append(step_dones.unsqueeze(1))

                end_number = self.max_length if training else self.num_times

                # end of one episode
                if step_dones.any() or len(list_step_states) > end_number:
                    truncated_indices = torch.nonzero(truncated.view(-1), as_tuple=False).view(-1)

                    # store the terminal state for truncated
                    t = len(list_step_states) - 1
                    for idx in truncated_indices:
                        idx = idx.item()
                        if idx in env_indices:  # check if this env is active
                            # update the action mask
                            actions_mask[idx, ...] = 0

                            term_up = step_infos[idx].get("terminal_observation", None)
                            if term_up is not None:
                                term_ts = util.to_ts(term_up, dtype=self.dtype, device=self.device)
                                terminal_record.append((t, idx + idx_per_env * num_env, term_ts))

                    # regenerate the dones_indices, because in training and testing, the dealing with dones is different
                    if training:
                        if len(list_step_states) > self.max_length:
                            dones_indices = torch.arange(num_env, dtype=torch.int64, device=self.device)
                        else:
                            dones_indices = torch.empty(0, dtype=torch.long, device=self.device)
                    else:
                        dones_indices = torch.nonzero(step_dones.view(-1), as_tuple=False).view(-1)

                    # if there's only one done environment, ensure dones_indices is a 1D tensor
                    if dones_indices.dim() == 0:
                        dones_indices = dones_indices.unsqueeze(0)

                    # process the data for the done env
                    for idx in dones_indices:
                        env_idx = idx.item()  # convert tensor to integer index
                        if env_idx in env_indices:
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
                                metric_value = util.get_item_from_dicts(step_infos, metric)[env_idx]
                                metric_value = util.to_ts(metric_value, self.dtype, self.device)
                                dict_task_specified_metrics[metric].append(metric_value)

                # update episode init state
                episode_init_state = step_states

                # for the next episode
                if len(env_indices) == 0:
                    break

                # clear the buffers for this env after processing
            list_step_states = list()
            list_step_actions = list()
            list_step_rewards = list()
            list_step_dones = list()

            # reset the env
            if not training:
                episode_init_state = envs.reset()
            else:
                self.train_env_init_state = episode_init_state
            # save the first state
            episode_init_state = util.to_ts(episode_init_state, dtype=self.dtype, device=self.device)
            list_step_states.append(episode_init_state.unsqueeze(1))

        # init the final results
        results = dict()

        if self.padding_on and not training:
            batch_size = len(list_step_states_result)  # this batch_size is actually the number of envs * ep_per_env
            max_sequence_length = self.num_times  # Desired sequence length after padding

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
                mask[:seq_length - 1] = True  # exclude the initial state
                masks.append(mask)

                # Helper function to pad a tensor
                def pad_tensor(tensor, pad_value=0, pad_length=max_sequence_length):
                    tensor_length, feature_length = tensor.size()
                    if tensor_length > pad_length:
                        tensor = tensor[:pad_length]
                    padded_tensor = torch.full(
                        (pad_length, feature_length),
                        pad_value,
                        dtype=tensor.dtype,
                        device=self.device,
                    )
                    padded_tensor[:tensor_length] = tensor
                    return padded_tensor

                # Pad each tensor in the sequence
                padded_states.append(pad_tensor(seq_states, pad_length=max_sequence_length + 1))  # init state
                padded_actions.append(pad_tensor(seq_actions))
                padded_rewards.append(pad_tensor(seq_rewards))
                padded_dones.append(pad_tensor(seq_dones, pad_value=1))  # 1 for done

            # stack the padded sequences and masks into batch tensors
            list_step_states_result = padded_states
            list_step_actions_result = padded_actions
            list_step_rewards_result = padded_rewards
            list_step_dones_result = padded_dones
            results["masks"] = torch.stack(masks, dim=0).unsqueeze(-1)
        else:  # no padding
            results["masks"] = torch.ones((len(list_step_states_result) * self.cut_segment,
                                           self.max_length // self.cut_segment, 1),
                                           dtype=torch.bool, device=self.device)

        # this tensor dones't take "terminal_observation" into account
        states_stack = torch.stack(list_step_states_result, dim=0)
        actions_stack = torch.stack(list_step_actions_result, dim=0)
        rewards_stack = torch.stack(list_step_rewards_result, dim=0)
        dones_stack = torch.stack(list_step_dones_result, dim=0)

        # Trim
        if training:
            B, T = states_stack.shape[0] , self.max_length
            states_stack = states_stack[:, :T+1, ...]
            actions_stack = actions_stack[:, :T, ...]
            rewards_stack = rewards_stack[:, :T, ...]
            dones_stack = dones_stack[:, :T, ...]

        num_total_env_steps = actions_stack.shape[:2].numel()

        # take "terminal_observation" into account
        if len(terminal_record) > 0 and training:
            record_idx = torch.zeros([B,], dtype=torch.long, device=self.device)
            zeros_actions = torch.zeros([1, policy.dim_out], dtype=self.dtype, device=self.device)
            zeros_rewards = torch.zeros([1, 1], dtype=self.dtype, device=self.device)
            zeros_dones = torch.zeros([1, 1], dtype=torch.bool, device=self.device)
            if training:
                for (timestep, env_idx, terminal_observation) in terminal_record:   # terminal_record was saved according to the original timestep, no sorting is needed
                    if timestep == T:
                        timestep += record_idx[env_idx]
                        record_idx[env_idx] += 1
                        states_stack[env_idx, timestep:, :] = torch.cat((terminal_observation.view(1, -1), states_stack[env_idx, timestep : -1, :]), dim=0)
                        actions_stack[env_idx, timestep:, :] = torch.cat((zeros_actions, actions_stack[env_idx, timestep : -1, :]), dim=0)
                        rewards_stack[env_idx, timestep:, :] = torch.cat((zeros_rewards, rewards_stack[env_idx, timestep : -1, :]), dim=0)
                        dones_stack[env_idx, timestep:, :] = torch.cat((zeros_dones, dones_stack[env_idx, timestep : -1, :]), dim=0)
                    else:
                        # truncation does not happen at the last step, so we need to reset the environment
                        # and discard this sample
                        self.train_env_init_state = envs.reset()
                        num_total_env_steps = 0

        else:   # TODO: theoretically, this one should be implemented, but currently not as it will not influence the
            # training, just the evaluation (negligible)
            pass

        if self.cut_segment > 1 and training:
            states_stack, actions_stack, rewards_stack, dones_stack = (
                self.segment_rollouts(states_stack, actions_stack, rewards_stack, dones_stack, self.cut_segment))

        # form up the results dict
        results["states"] = states_stack
        results["actions"] = actions_stack
        results["rewards"] = rewards_stack
        results["dones"] = dones_stack

        if self.task_specified_metrics is not None:
            for metric in self.task_specified_metrics:
                results[metric] = torch.stack(dict_task_specified_metrics[metric], dim=0)

        return results, num_total_env_steps

    @staticmethod
    def segment_rollouts(states_stack, actions_stack, rewards_stack, dones_stack, S: int):
        """
        Split B-length rollouts of length T (states length T+1) into S equal segments,
        then fold segments into the batch dimension.

        states_stack:  [B, T+1, Fs]
        actions_stack: [B, T,   Fa]
        rewards_stack: [B, T]   or [B, T, 1]
        dones_stack:   [B, T]   or [B, T, 1]

        Returns:
        states_out:    [B*S, T/S + 1, Fs]
        actions_out:   [B*S, T/S,     Fa]
        rewards_out:   [B*S, T/S,     1 ]
        dones_out:     [B*S, T/S,     1 ]
        """
        if states_stack.ndim != 3 or actions_stack.ndim != 3:
            raise ValueError("states_stack and actions_stack must be 3D: [B, T(+1), F].")

        B, Tp1, Fs = states_stack.shape
        B2, T, Fa = actions_stack.shape
        if B2 != B or Tp1 != T + 1:
            raise ValueError(f"Shape mismatch: states have T+1={Tp1}, actions have T={T} (need Tp1 == T+1), and B must match.")

        if T <= 0:
            raise ValueError("T must be > 0.")
        if S <= 0:
            raise ValueError("S must be > 0 (number of segments).")
        if T % S != 0:
            raise ValueError(f"T ({T}) must be divisible by S ({S}).")

        # Normalize rewards/dones to [B, T, 1]
        if rewards_stack.ndim == 2:
            rewards_stack = rewards_stack.unsqueeze(-1)
        if dones_stack.ndim == 2:
            dones_stack = dones_stack.unsqueeze(-1)

        if rewards_stack.shape[:2] != (B, T) or rewards_stack.shape[-1] != 1:
            raise ValueError("rewards_stack must be [B, T, 1] or [B, T].")
        if dones_stack.shape[:2] != (B, T) or dones_stack.shape[-1] != 1:
            raise ValueError("dones_stack must be [B, T, 1] or [B, T].")

        L = T // S  # steps per segment

        # --- Actions / rewards / dones: split evenly along time and fold (B,S)->(B*S)
        actions_out = actions_stack.contiguous().view(B, S, L, Fa).reshape(B * S, L, Fa)
        rewards_out = rewards_stack.contiguous().view(B, S, L, 1).reshape(B * S, L, 1)
        dones_out   = dones_stack.contiguous().view(B, S, L, 1).reshape(B * S, L, 1)

        # --- States: need L+1 per segment. Use unfold(size=L+1, step=L) along time.
        # Result of unfold on dim=1: [B, S, L+1, Fs] -> then merge (B,S)->(B*S)
        states_unfold = states_stack.unfold(dimension=1, size=L + 1, step=L).permute(0, 1, 3, 2)  # [B, S, L+1, Fs]
        # Make memory contiguous before view (unfold often yields a non-contiguous view)
        states_out = states_unfold.contiguous().view(B * S, L + 1, Fs)

        return states_out, actions_out, rewards_out, dones_out