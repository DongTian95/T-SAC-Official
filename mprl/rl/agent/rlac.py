import os
import sys
from collections import namedtuple

import numpy as np
import torch
import random
import copy

import mprl.util as util
from mprl.rl.agent import AbstractAgent

import mprl.rl.critic.transformer_sac_critic as transformer_sac_critic
import mprl.rl.policy.transformer_sac_policy as transformer_sac_policy
import mprl.rl.sampler.transformer_sac_sampler as transformer_sac_sampler
import mprl.rl.replay_buffer.transformer_sac_replay_buffer as transformer_sac_rb

class RLAC(AbstractAgent):
    def __init__(
            self,
            policy: transformer_sac_policy.TransformerSACPolicy,
            critic: transformer_sac_critic.TransformerSACCritic,
            sampler: transformer_sac_sampler.TransformerSACSampler,
            replay_buffer: transformer_sac_rb.TransformerSACReplayBuffer,
            projection=None,
            dtype=torch.float32,
            device=torch.device("cpu"),
            **kwargs,
    ):
        super().__init__(policy, critic, sampler, dtype=dtype, device=device, **kwargs)

        # for Adam optimizer
        self.betas = kwargs.get("betas")
        if isinstance(self.betas, str):     # deal with betas being a string
            self.betas = tuple(map(float, self.betas.strip('()').split(',')))

        self.policy_betas = kwargs.get("policy_betas", self.betas)
        if isinstance(self.policy_betas, str):     # deal with betas being a string
            self.policy_betas = tuple(map(float, self.policy_betas.strip('()').split(',')))

        self.critic_betas = kwargs.get("critic_betas", self.betas)
        if isinstance(self.critic_betas, str):     # deal with betas being a string
            self.critic_betas = tuple(map(float, self.critic_betas.strip('()').split(',')))

        self.clip_critic = float(kwargs.get("clip_critic"))
        self.clip_grad_norm = float(kwargs.get("clip_grad_norm"))

        self.balance_check = kwargs.get("balance_check")
        self.evaluation_interval = kwargs.get("evaluation_interval")

        # network initialization
        self.policy = policy
        self.critic = critic
        self.sampler = sampler
        self.replay_buffer = replay_buffer

        self.use_automatic_entropy_tuning = kwargs.get("use_automatic_entropy_tuning", False)
        self.entropy_penalty_coef = torch.nn.Parameter(
            torch.tensor(kwargs.get("entropy_penalty_coef"), device=self.device, dtype=self.dtype))
        if self.use_automatic_entropy_tuning:
            self.temperature_optimizer = torch.optim.AdamW(
                [self.entropy_penalty_coef], lr=self.lr_policy, weight_decay=self.wd_policy, betas=self.betas
            )

        self.target_entropy = kwargs.get("target_entropy", "default")
        if self.target_entropy == "default":
            self.target_entropy = -np.prod(self.sampler.debug_env.envs[0].action_space.shape)

        # Optimizers initialization
        self.separate_mean_variance_optimizer = kwargs.get("separate_mean_variance_optimizer", False)
        if not self.separate_mean_variance_optimizer:
            self.policy_optimizer = self.get_policy_optimizer(self.policy)
        else:
            # in this case, self.policy_optimizer is a list
            self.policy_optimizer = self.get_separate_policy_optimizer(self.policy,
                                                                       scaling=self.separate_mean_variance_optimizer)
        self.critic_optimizer = self.get_critic_optimizer(self.critic)

        # Learning rate schedulers
        self.policy_lr_scheduler, self.critic_lr_scheduler = \
            self.get_lr_scheduler()

        # for AMP (Automatic Mixed Precision)
        # self.scaler = torch.amp.GradScaler()

        # special for Transformer SAC
        self.critic_warmup_step = kwargs.get("critic_warmup_step")
        self.policy_warmup_step = kwargs.get("policy_warmup_step")
        self.training_times = kwargs.get("training_times")

        self.random_critic_index_number = kwargs.get("random_critic_index_number")
        if self.random_critic_index_number == "default":
            self.random_critic_index_number = self.sampler.debug_env.envs[0].spec.max_episode_steps
        self.random_policy_index_number = kwargs.get("random_policy_index_number")

        self.random_target = kwargs.get("random_target")

        # critic training in order
        self.critic_train_in_order = kwargs.get("critic_train_in_order", False)

        # whether add timestamp
        self.add_timestamp = kwargs.get("add_timestamp", True)
        
        # whether using dones to set the target value to 0
        self.use_done_for_target = kwargs.get("use_done_for_target", True)

        # punish deviation between critic output
        self.punish_deviation = kwargs.get("punish_deviation", None)

        # separate policy and entropy training
        self.separate_policy_entropy_training = kwargs.get("separate_policy_entropy_training", False)
        if self.separate_policy_entropy_training:
            self.entropy_optimizer = self.get_entropy_optimizer(self.policy)
            if int(self.separate_policy_entropy_training) != 1:
                self.entropy_update_index = int(self.separate_policy_entropy_training)

        # target net update interval
        self.target_update_interval = kwargs.get("target_update_interval", 1)
        self.target_update_index = int(self.target_update_interval)

        # get the trust region setting
        self.projection = projection
        self.use_trust_region_update = kwargs.get("use_trust_region_update", False)
        if self.use_trust_region_update:
            self.old_policy = copy.deepcopy(self.policy)
            self.old_policy.eval()
            self.use_last_three_as_policy_target = kwargs.get("use_last_three_as_policy_target", False)
            if self.use_last_three_as_policy_target:
                self.last_policy = copy.deepcopy(self.policy)
                self.last_policy.eval()
                self.second_last_policy = copy.deepcopy(self.policy)
                self.second_last_policy.eval()
                self.third_last_policy = copy.deepcopy(self.policy)
                self.third_last_policy.eval()
            self.policy_polyak_average_factor = kwargs.get("policy_polyak_average_factor", 1.0)
            self.set_variance = kwargs.get("set_variance", False)

        self.huber_beta = kwargs.get("huber_beta", None)
        self.policy_update_index = 0

    def get_separate_policy_optimizer(self, policy, scaling: int = 1):
        mean_net_params = policy.mean_net.parameters()
        variance_net_params = policy.variance_net.parameters()
        mean_net_optimizer = torch.optim.AdamW(params=mean_net_params, lr=self.lr_policy,
                                               weight_decay=self.wd_policy, betas=self.policy_betas)
        variance_net_optimizer = torch.optim.AdamW(params=variance_net_params, lr=self.lr_policy * scaling,
                                                   weight_decay=self.wd_policy, betas=self.policy_betas)
        return [mean_net_optimizer, variance_net_optimizer]

    def get_entropy_optimizer(self, policy):
        variance_net_params = policy.variance_net.parameters()
        entropy_optimizer = torch.optim.AdamW(params=variance_net_params, lr=self.lr_policy,
                                              weight_decay=self.wd_policy, betas=self.policy_betas)
        return entropy_optimizer

    def get_policy_optimizer(self, policy):
        # policy optimizer
        self.policy_net_params = policy.parameters

        def _iter_policy_params(policy):
            for attr in ("mean_net", "variance_net"):
                net = getattr(policy, attr, None)
                if net is None:
                    continue
                params_attr = getattr(net, "parameters", None)
                if callable(params_attr):
                    for p in params_attr():
                        yield p
                elif params_attr is not None:
                    for p in params_attr:
                        yield p
                v = getattr(net, "variable", None)
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    yield v

        # split parameters: decay vs no_decay (LayerNorm + biases typically no decay)
        decay, no_decay = [], []
        for p in _iter_policy_params(policy):
            if not p.requires_grad:
                continue
            (no_decay if p.dim() < 2 else decay).append(p)

        groups = [
            {"params": decay, "weight_decay": self.wd_policy},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        # Optional: keep a snapshot of params
        self.policy_net_params = list(_iter_policy_params(policy))

        return torch.optim.AdamW(groups, lr=self.lr_policy, betas=self.policy_betas)

    def get_critic_optimizer(self, critic):
        self.critic_net_params = {
            "net1": list(critic.net1.parameters()),  # Added parentheses
            "net2": list(critic.net2.parameters())  # Added parentheses
        }

        critic_opt1, critic_opt2 = self.critic.configure_optimizer(
            weight_decay=self.wd_critic,
            learning_rate=self.lr_critic,
            betas=self.critic_betas,
        )

        return [critic_opt1, critic_opt2]

    def step(self):
        # update total step count
        self.num_iterations += 1

        # collect dataset
        util.run_time_test(lock=True)
        dataset, num_env_interaction = self.sampler.run(training=True, policy=self.policy, critic=self.critic)
        self.num_global_steps += num_env_interaction
        sampling_time = util.run_time_test(lock=False)

        # save dataset to RB
        util.run_time_test(lock=True)
        if self._check_sampler_dataset(dataset):    # only add to RB if the dataset is valid
            self.replay_buffer.add(dataset)
        else:
            print("Discard dataset due to NaN, Inf or large values")
        dataset_stats = util.generate_many_stats(dataset, "exploration", to_np=True)
        save_to_rb = util.run_time_test(lock=False)

        # Update agent
        if self.num_iterations > self.critic_warmup_step and self.replay_buffer.batch_size <= len(self.replay_buffer):
            # critic warmup phase and RB has enough data
            util.run_time_test(lock=True, key="update")

            for idx in range(self.training_times_critic):

                critic_dataset = self.replay_buffer.prioritized_sample()

                util.run_time_test(lock=True, key="update critic")
                critic_loss_dict = self.update_critic(critic_dataset)
                if self.schedule_lr_critic:
                    self.critic_lr_scheduler[0].step()
                    self.critic_lr_scheduler[1].step()
                update_critic_time = util.run_time_test(lock=False, key="update critic")

            if self.num_iterations <= self.policy_warmup_step:  # warm up phase for policy
                update_time = util.run_time_test(lock=False, key="update")
                result_metrics = {
                    **dataset_stats, **critic_loss_dict,
                    "sampling_time": sampling_time,
                    "update_time": update_time,
                    "update_critic_time": update_critic_time,
                    "update_policy_time": 0.0,
                    "save_to_rb": save_to_rb,
                    "num_global_steps": self.num_global_steps,
                    "lr_policy": self.policy_lr_scheduler.get_last_lr()[
                        0] if self.schedule_lr_policy else self.lr_policy,
                    "lr_critic": self.critic_lr_scheduler[0].get_last_lr()
                        if self.schedule_lr_critic else self.lr_critic,
                }
            else:   # update policy
                util.run_time_test(lock=True, key="update policy")
                for _ in range(self.training_times_policy):
                    policy_dataset = self.replay_buffer.sample()
                    policy_loss_dict = self.update_policy(policy_dataset)
                    if self.schedule_lr_policy:
                        self.policy_lr_scheduler.step()
                update_policy_time = util.run_time_test(lock=False, key="update policy")

                update_time = util.run_time_test(lock=False, key="update")

                result_metrics = {
                    **dataset_stats, **critic_loss_dict, **policy_loss_dict,
                    "sampling_time": sampling_time,
                    "update_time": update_time,
                    "update_critic_time": update_critic_time,
                    "update_policy_time": update_policy_time,
                    "save_to_rb": save_to_rb,
                    "num_global_steps": self.num_global_steps,
                    "lr_policy": self.policy_lr_scheduler.get_last_lr()[0] if self.schedule_lr_policy else self.lr_policy,
                    "lr_critic": self.critic_lr_scheduler[0].get_last_lr() if self.schedule_lr_critic else self.lr_critic,
                }

            # Evaluate agent
            if self.evaluation_interval == 1 or self.num_iterations % self.evaluation_interval == 1:
                util.run_time_test(lock=True)
                evaluate_metrics = util.generate_many_stats(self.evaluate()[0], "evaluation", to_np=True)
                evaluation_time = util.run_time_test(lock=False)
                result_metrics.update(evaluate_metrics)
                result_metrics.update({"evaluation_time": evaluation_time})

            return result_metrics

        # No enough samples in RB
        return {
            **dataset_stats,
            "sampling_time": sampling_time,
            "update_time": 0.0,
            "update_critic_time": 0.0,
            "update_policy_time": 0.0,
            "save_to_rb": save_to_rb,
            "num_global_steps": self.num_global_steps,
            "lr_policy": self.policy_lr_scheduler.get_last_lr()[0] if self.schedule_lr_policy else self.lr_policy,
            "lr_critic": self.critic_lr_scheduler[0].get_last_lr() if self.schedule_lr_critic else self.lr_critic,
        }   # No update done, because the warmup phase for critic is not finished

    def update_critic(self, dataset: dict):
        # data extraction
        states = dataset["states"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        dones = dataset["dones"]
        masks = dataset["masks"]

        # get important parameters
        batch_size, sample_length, _ = actions.shape
        min_length = self.sampler.min_length
        step_length = self.sampler.step_length
        action_range = self.sampler.action_range

        # critic loss saver
        critic_loss_raw = []
        critic_grad_norm = []
        clipped_critic_grad_norm = []

        # target value saver
        target_values1, target_values2 = [], []
        target_values_list = []
        mc_returns_list = []
        target_mc_bias_list = []

        # get the longest trajectory
        longest_length = sample_length

        # initialize the loop
        if self.critic_train_in_order == "segment":
            random_indices = self.partition_trajectory_with_optional_discard(
                n = longest_length, max_len=step_length, min_len=min_length,
                num_segments=self.random_critic_index_number, discard_allowed=False)
        elif self.critic_train_in_order:  # maybe better for sparse
            random_indices = range(longest_length - 1, -1, -1)
        else:   # better for dense
            # longest_length = longest_length - min_length
            if 0 < self.random_critic_index_number < 1:     # implemented as percentage
                num_indices = int(self.random_critic_index_number * longest_length)
                if num_indices < 1:     # deal with num_indices == 0
                    num_indices = 1
                random_indices = random.sample(range(longest_length), num_indices)
            else:
                num_indices = longest_length if longest_length < self.random_critic_index_number \
                    else self.random_critic_index_number
                random_indices = random.sample(range(longest_length), num_indices)

        # calculate the target values
        # Attention: T-SAC calculates the target value of the whole trajectory at once, in one loop,
        # updating the target net will NOT lead to the recalculation of the target values
        target1, target2 = self.get_target_values(states, start_idx=1)

        # critic_loss
        # TODO: this part is horrible, need to be refactored, DON NOT mix list and dict
        for start_idx in random_indices:
            if self.critic_train_in_order == "segment":
                end_idx = random_indices[start_idx]
                rand_int = end_idx - start_idx
            else:
                rand_int = torch.randint(min_length, step_length + 1, (1,)).item()   # to determine the start index
                # rand_int = step_length
                end_idx = start_idx + rand_int
                # update from sample_length to longest_length, TODO: seems no influence
                if end_idx > longest_length:   # reached the beginning of the trajectory
                    end_idx = longest_length
                    rand_int = end_idx - start_idx

            ######################### get the target value #########################
            # target_states = states[..., start_idx: end_idx + 1, :]
            # c_target1, c_target2 = self.get_target_values(target_states, start_idx=start_idx + 1)
            c_target1, c_target2 = target1[..., start_idx:end_idx], target2[..., start_idx:end_idx]

            # get the start state infos
            c_state = states[..., start_idx, :]
            c_actions = actions[..., start_idx: end_idx, :]
            c_rewards = rewards[..., start_idx: end_idx, :]
            c_dones = dones[..., start_idx: end_idx, :]
            # c_masks = masks[..., start_idx: end_idx, :].view(batch_size, -1)

            c_masks = torch.ones((batch_size, rand_int, 1), dtype=torch.bool, device=self.device)
            d = (c_dones == 1)
            kill = d.cumsum(dim=1) > 0
            kill_after = torch.cat(
                [torch.zeros_like(kill[:, :1, ...], dtype=torch.bool), kill[:, :-1, ...]],
                dim=1
            )
            c_masks = c_masks.masked_fill(kill_after, 0).view(batch_size, -1)


            if self.add_timestamp == "only_action":
                # add time stamp to c_actions
                c_actions_time_stamp = torch.arange(start_idx, end_idx, device=self.device, dtype=self.dtype)
                c_actions_time_stamp = c_actions_time_stamp.repeat(batch_size, 1)
                c_actions_time_stamp = c_actions_time_stamp.view(batch_size, -1, 1)
                c_actions = torch.cat((c_actions_time_stamp, c_actions), dim=-1)
            elif self.add_timestamp == "only_state":
                # add time stamp to c_state
                c_state_time_stamp = util.to_ts(start_idx, self.dtype, self.device)
                c_state_time_stamp = c_state_time_stamp.repeat(batch_size, 1)
                c_state = torch.cat((c_state_time_stamp, c_state), dim=-1)
            elif self.add_timestamp:
                # add time stamp to c_state
                c_state_time_stamp = util.to_ts(start_idx, self.dtype, self.device)
                c_state_time_stamp = c_state_time_stamp.repeat(batch_size, 1)
                c_state = torch.cat((c_state_time_stamp, c_state), dim=-1)

                # add time stamp to c_actions
                c_actions_time_stamp = torch.arange(start_idx, end_idx, device=self.device, dtype=self.dtype)
                c_actions_time_stamp = c_actions_time_stamp.repeat(batch_size, 1)
                c_actions_time_stamp = c_actions_time_stamp.view(batch_size, -1, 1)
                c_actions = torch.cat((c_actions_time_stamp, c_actions), dim=-1)

            # whether using random or min to get the target
            if self.random_target == "mean":
                c_target = (c_target1 + c_target2) / 2
            elif self.random_target:
                # randomly choose the Q value from Q1 or Q2
                target_mask = torch.randint(0, 2, c_target1.shape, device=self.device)
                c_target = target_mask * c_target1 + (1 - target_mask) * c_target2
            else:
                c_target = torch.min(c_target1, c_target2)

            # considering the discount factor
            discounts_factors = torch.tensor([self.discount_factor ** i for i in range(1, rand_int + 1)],
                                             device=self.device, dtype=self.dtype)
            discounts_factors = discounts_factors.repeat(batch_size, 1)
            c_target = c_target * discounts_factors

            # considering the dones or masks
            c_target = c_target * (1 - c_dones.view(batch_size, -1))

            #####################for report#######################
            # # get Monte-Car;p return
            # mc_returns = torch.sum(rewards[..., end_idx : , :], dim=1)
            # mc_returns = mc_returns.view(batch_size, -1)
            #
            # # get the target Monte_Carlo bias
            # mc_bias = c_target[..., -1] - mc_returns
            #
            # save the infos
            target_value = util.to_np(c_target.mean())
            target_values_list.append(target_value)

            # mc_returns = util.to_np(mc_returns.mean())
            # mc_returns_list.append(mc_returns)
            #
            # mc_bias = util.to_np(mc_bias.mean())
            # target_mc_bias_list.append(mc_bias)

            ###################for report part end################

            # add the discounted reward
            discounted_rewards = torch.zeros_like(c_rewards)
            discounted_rewards[..., 0, :] = c_rewards[..., 0, :]
            for idx in range(1, rand_int):
                discounted_rewards[..., idx, :] = (discounted_rewards[..., idx - 1, :] +
                                                   self.discount_factor ** idx * c_rewards[..., idx, :])

            c_target += discounted_rewards.view(batch_size, -1)
            c_target = c_target.detach()    # just for sure

            ######################### get the target value end #####################

            # get the current value
            self.critic_optimizer[0].zero_grad(set_to_none=True)
            self.critic_optimizer[1].zero_grad(set_to_none=True)

            # with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32, enabled=False):
            # generate the idx_c and idx_a tensors
            # idx_c, idx_a = self.critic.generate_idx_tensors(
            #     batch_size=batch_size,
            #     sequence_length=sample_length,
            #     current_states_idx=start_idx,
            #     step_length=rand_int
            # )
            c_actions = c_actions * c_masks.unsqueeze(-1)

            # get the current values
            c_current1 = self.critic.critic(net=self.critic.net1, c_state=c_state, actions=c_actions,
                                            idx_c=None, idx_a=None, no_absolute_idx=True)
            c_current2 = self.critic.critic(net=self.critic.net2, c_state=c_state, actions=c_actions,
                                            idx_c=None, idx_a=None, no_absolute_idx=True)

            # take the mask
            c_target = c_target * c_masks

            c_current1 = c_current1 * c_masks
            c_current2 = c_current2 * c_masks

            # compute the loss
            # c_loss1 = torch.nn.functional.mse_loss(c_current1, c_target)
            # c_loss2 = torch.nn.functional.mse_loss(c_current2, c_target)
            if self.huber_beta is None:
                c_loss1 = torch.nn.functional.mse_loss(c_current1, c_target, reduction="none")
            else:
                c_loss1 = torch.nn.functional.smooth_l1_loss(c_current1, c_target, reduction='none', beta=self.huber_beta)
            c_loss1 = c_loss1 * c_masks
            # c_loss1 = c_loss1.sum(-1) / c_masks.sum(-1)
            c_loss1 = c_loss1.mean()
            # c_loss1 = (c_loss1 * c_masks).mean()
            # c_loss1 = c_loss1 * c_masks
            # c_loss1 = c_loss1.sum() / c_masks.sum()
            if self.huber_beta is None:
                c_loss2 = torch.nn.functional.mse_loss(c_current2, c_target, reduction="none")
            else:
                c_loss2 = torch.nn.functional.smooth_l1_loss(c_current2, c_target, reduction='none', beta=self.huber_beta)
            c_loss2 = c_loss2 * c_masks
            # c_loss2 = c_loss2.sum(-1) / c_masks.sum(-1)
            c_loss2 = c_loss2.mean()


            # the difference of the outputs of critic net shouldn't be significant
            if self.punish_deviation is not None and c_current1.shape[-1] > 1:
                c_loss1 += self.punish_deviation * c_current1.std(dim=1).mean()
                c_loss2 += self.punish_deviation * c_current2.std(dim=1).mean()
            
            # backpropagation of the critic loss
            # self.scaler.scale(c_loss1).backward()
            # self.scaler.scale(c_loss2).backward()
            # c_loss1.backward()
            # c_loss2.backward()
            (c_loss1 + c_loss2).backward()

            if self.clip_grad_norm > 0:
                grad_norm_1 = torch.nn.utils.clip_grad_norm_(parameters=self.critic_net_params["net1"],
                                               max_norm=self.clip_grad_norm, error_if_nonfinite=False)
                grad_norm_c_1  = torch.nn.utils.clip_grad_norm_(parameters=self.critic_net_params["net1"],
                                               max_norm=float('inf'), error_if_nonfinite=False)
                grad_norm_2 = torch.nn.utils.clip_grad_norm_(parameters=self.critic_net_params["net2"],
                                               max_norm=self.clip_grad_norm, error_if_nonfinite=False)
                grad_norm_c_2 = torch.nn.utils.clip_grad_norm_(parameters=self.critic_net_params["net2"],
                                               max_norm=float('inf'), error_if_nonfinite=False)

                # store the gradient norms and loss
                critic_grad_norm.append(util.to_np(grad_norm_1))
                clipped_critic_grad_norm.append(util.to_np(grad_norm_c_1))
                critic_grad_norm.append(util.to_np(grad_norm_2))
                clipped_critic_grad_norm.append(util.to_np(grad_norm_c_2))
            else:
                grad_norm_1 = torch.nn.utils.clip_grad_norm_(parameters=self.critic_net_params["net1"],
                                               max_norm=float('inf'), error_if_nonfinite=False)
                grad_norm_2 = torch.nn.utils.clip_grad_norm_(parameters=self.critic_net_params["net2"],
                                               max_norm=float('inf'), error_if_nonfinite=False)

                critic_grad_norm.append(util.to_np(grad_norm_1))
                clipped_critic_grad_norm.append(util.to_np(grad_norm_1))
                critic_grad_norm.append(util.to_np(grad_norm_2))
                clipped_critic_grad_norm.append(util.to_np(grad_norm_2))

            critic_loss_raw.append(c_loss1.item())
            critic_loss_raw.append(c_loss2.item())
            
            # update the critic
            # self.scaler.step(self.critic_optimizer[0])
            # self.scaler.step(self.critic_optimizer[1])
            self.critic_optimizer[0].step()
            self.critic_optimizer[1].step()

            # update the scaler
            # self.scaler.update()

            # polyak update target network
            self.target_update_index -= 1
            if self.target_update_index <= 0:
                self._soft_copy(target_model=self.critic.target_net1, source_model=self.critic.net1)
                self._soft_copy(target_model=self.critic.target_net2, source_model=self.critic.net2)
                self.target_update_index = int(self.target_update_interval)

        return {
            **util.generate_stats(critic_loss_raw, name="critic_loss"),
            **util.generate_stats(critic_grad_norm, name="critic_grad_norm"),
            **util.generate_stats(clipped_critic_grad_norm, name="clipped_critic_grad_norm"),
            **util.generate_stats(target_values_list, name="target_values"),
            # **util.generate_stats(mc_returns_list, name="mc_returns"),
            # **util.generate_stats(target_mc_bias_list, name="target_mc_bias"),
        }

    def update_policy(self, dataset: dict):
        # data extraction
        states = dataset["states"]
        dones = dataset["dones"]
        masks = dataset["masks"]

        # get some parameters
        batch_size, sample_length, _ = states.shape
        sample_length -= 1  # last state is the done state, no action will be taken at that step
        action_range = self.sampler.action_range

        # loss storage
        policy_loss_raw = []
        entropy_loss_raw = []
        surrogate_loss_raw = []

        # entropy storage
        entropy = []

        # gradient storage
        policy_grad_norm = []
        clipped_policy_grad_norm = []

        # generate some idx for updating

        # get the longest trajectory
        longest_length = sample_length

        # get the num_indices
        if isinstance(self.random_policy_index_number, list):
            lower_bound = self.random_policy_index_number[0]
            upper_bound = self.random_policy_index_number[1]
        elif self.random_policy_index_number >= 1:     # implemented as a absolute number
            num_indices = longest_length if longest_length < self.random_policy_index_number \
                else self.random_policy_index_number
        else:   # implemented as a ratio
            num_indices = int(longest_length * self.random_policy_index_number)
            if num_indices < 1:     # to deal with num_indices = 0
                num_indices = 1

        if isinstance(self.random_policy_index_number, list):
            random_interval = random.randint(lower_bound, upper_bound)
            # Number of intervals that fit in sample_length
            num_indices = sample_length // random_interval

            # Ensure we have at least 1 index
            if num_indices < 1:
                # Decide how to handle this scenario, e.g. set num_indices = 1 or raise an error.
                num_indices = 1

            # Make sure the last index doesn't exceed sample_length - 1
            #   last_index = start_index + (num_indices - 1)*random_interval
            #   => start_index <= sample_length - 1 - (num_indices - 1)*random_interval
            max_start_index = sample_length - 1 - (num_indices - 1) * random_interval

            # If no valid start is possible, decide how to handle.
            if max_start_index < 0:
                # For example: clamp to 0 or reduce random_interval or handle error.
                max_start_index = 0

            start_index = random.randint(0, max_start_index)

            random_indices = [start_index + i * random_interval for i in range(num_indices)]
        else:
            random_indices = random.sample(range(longest_length), num_indices)

        # loop over the trajectories
        for start_idx in random_indices:
            if not self.separate_mean_variance_optimizer:
                self.policy_optimizer.zero_grad(set_to_none=True)
            else:
                self.policy_optimizer[0].zero_grad(set_to_none=True)
                self.policy_optimizer[1].zero_grad(set_to_none=True)

            # with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32, enabled=False):
            # get the current state info
            c_state = states[:, start_idx, :]
            c_masks = masks[:, start_idx, :]

            if self.add_timestamp == "only_action":
                pass
            elif self.add_timestamp:
                # add time stamp to next_state
                c_state_time_stamp = util.to_ts(start_idx, self.dtype, self.device)
                c_state_time_stamp = c_state_time_stamp.repeat(batch_size, 1)
                c_state = torch.cat((c_state_time_stamp, c_state), dim=-1)

            # get trust region loss
            if self.use_trust_region_update:
                params_mean, params_L, trust_region_loss = self.trust_region_update(c_state) # Caution: not include add time stamp to action
            else:
                # generate the params_mean and params_std
                params_mean, params_L = self.policy.policy(obs=c_state, action_range=action_range)

            # sample the action
            actions = self.policy.sample(params_mean=params_mean, params_L=params_L, require_grad=True)

            # update the entropy penalty coefficient
            if self.use_automatic_entropy_tuning:
                self.policy_update_index += 1
                if self.policy_update_index >= 10000:
                    if self.separate_policy_entropy_training:
                        if int(self.separate_policy_entropy_training) != 1:
                            if self.entropy_update_index > 0:
                                continue
                    log_probs = self.policy.log_prob(smp_params=actions, params_mean=params_mean, params_L=params_L)
                    alpha = torch.nn.functional.softplus(self.entropy_penalty_coef)    # ensures alpha is always positive
                    alpha_loss = -(alpha * (log_probs + self.target_entropy).detach()).mean()

                    self.temperature_optimizer.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    self.temperature_optimizer.step()

            actions = torch.tanh(actions) * action_range

            # entropy penalty loss
            entropy_loss, entropy_stats = self.entropy_loss(params_mean=params_mean, params_L=params_L)

            if self.add_timestamp == "only_state":
                pass
            elif self.add_timestamp:
                # add time stamp to actions
                actions_time_stamp = util.to_ts(start_idx, self.dtype, self.device)
                actions_time_stamp = actions_time_stamp.repeat(batch_size, 1).view(batch_size, -1, 1)
                actions = torch.cat((actions_time_stamp, actions), dim=-1)

            # get the idx for critic networks
            # idx_c, idx_a = self.critic.generate_idx_tensors(
            #     batch_size=batch_size,
            #     sequence_length=sample_length,
            #     current_states_idx=start_idx,
            #     step_length=1,
            # )

            # get the Q values
            q_1 = self.critic.critic(net=self.critic.net1, c_state=c_state, actions=actions,
                                     idx_c=None, idx_a=None, no_absolute_idx=True)
            q_2 = self.critic.critic(net=self.critic.net2, c_state=c_state, actions=actions,
                                     idx_c=None, idx_a=None, no_absolute_idx=True)
            q_value = torch.min(q_1, q_2)
            # q_value = q_1

            # get the surrogate loss
            surrogate_loss = -q_value

            # get the policy loss
            # policy_loss = surrogate_loss + entropy_loss
            if self.separate_policy_entropy_training:
                policy_loss = surrogate_loss
            else:
                policy_loss = surrogate_loss + entropy_loss

            # add trust region loss
            if self.use_trust_region_update:
                policy_loss = policy_loss + trust_region_loss

            # take the mask into account
            policy_loss = policy_loss * c_masks.view(batch_size)

            policy_loss = policy_loss.mean()

            # backward
            # self.scaler.scale(policy_loss).backward()
            policy_loss.backward()

            if self.clip_grad_norm > 0:
                # clip the gradient (the util.grad_norm_clip is too slow)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net_params,
                                                      max_norm=self.clip_grad_norm, error_if_nonfinite=False)
                grad_norm_c = torch.nn.utils.clip_grad_norm_(self.policy_net_params,
                                                      max_norm=float('inf'), error_if_nonfinite=False)
                policy_grad_norm.append(util.to_np(grad_norm))
                clipped_policy_grad_norm.append(util.to_np(grad_norm_c))
            else:
                # just save the norm
                grad_norm = util.to_np(torch.nn.utils.clip_grad_norm_(self.policy_net_params,
                                                      max_norm=float('inf'), error_if_nonfinite=False))
                policy_grad_norm.append(util.to_np(grad_norm))
                clipped_policy_grad_norm.append(util.to_np(grad_norm))

            # clip the gradient
            # grad_norm, grad_norm_c = util.grad_norm_clip(bound=self.clip_grad_norm,
            #                                              params=self.policy_net_params)

            # logging
            surrogate_loss_raw.append(surrogate_loss.mean().item())
            entropy_loss_raw.append(entropy_loss.item())
            policy_loss_raw.append(policy_loss.item())
            entropy.append(entropy_stats["entropy"])

            # update the policy
            # self.scaler.step(self.policy_optimizer)
            if not self.separate_mean_variance_optimizer:
                self.policy_optimizer.step()
            else:
                self.policy_optimizer[0].step()
                self.policy_optimizer[1].step()

            if self.separate_policy_entropy_training:
                if int(self.separate_policy_entropy_training) != 1:
                    if self.entropy_update_index > 0:
                        self.entropy_update_index -= 1
                        continue
                    else:
                        self.entropy_update_index = int(self.separate_policy_entropy_training)
                params_mean, params_L = self.policy.policy(obs=c_state, action_range=action_range)
                # entropy penalty loss
                entropy_loss, entropy_stats = self.entropy_loss(params_mean=params_mean, params_L=params_L)
                entropy_loss = entropy_loss * c_masks.view(batch_size)
                entropy_loss = entropy_loss.mean()
                self.entropy_optimizer.zero_grad(set_to_none=True)
                entropy_loss.backward()
                self.entropy_optimizer.step()

            # update the scaler
            # self.scaler.update()

        return {
            **util.generate_stats(policy_loss_raw, name="policy_loss"),
            **util.generate_stats(entropy_loss_raw, name="entropy_loss"),
            **util.generate_stats(surrogate_loss_raw, name="surrogate_loss"),
            **util.generate_stats(entropy, name="entropy"),
            **util.generate_stats(policy_grad_norm, name="policy_grad_norm"),
            **util.generate_stats(clipped_policy_grad_norm, name="clipped_policy_grad_norm"),
        }
    
    @torch.no_grad()
    def get_target_values(self, states, start_idx=1):
        # the first state doesn't need target value
        states = states[:, 1:, :].clone()
        # states = states[:, 1:, :]

        # get important parameters
        batch_size, sample_length, feature_length = states.shape

        if self.add_timestamp == "only_action":
            pass
        elif self.add_timestamp:
            # generate some idx for updating
            indices_tensor = torch.arange(start_idx, sample_length + start_idx, dtype=self.dtype, device=self.device)
            indices_tensor = indices_tensor.repeat(batch_size, 1)
            states = torch.cat((indices_tensor.unsqueeze(-1), states), dim=-1)

        # to boost the calculation
        states = states.view(batch_size * sample_length, -1)
        params_mean, params_L = self.policy.policy(obs=states, action_range=self.sampler.action_range)

        # sample the action
        pred_actions = self.policy.sample(params_mean=params_mean, params_L=params_L,
                                          use_mean=False, require_grad=False)
        pred_actions = torch.tanh(pred_actions) * self.sampler.action_range

        if self.add_timestamp == "only_state":
            pass
        elif self.add_timestamp:
            # add time stamp to pred_actions
            indices_tensor = torch.arange(start_idx, sample_length + start_idx, dtype=self.dtype,
                                          device=self.device)
            indices_tensor = indices_tensor.repeat(batch_size, 1)
            indices_tensor = indices_tensor.view(batch_size * sample_length, -1)
            pred_actions = torch.cat((indices_tensor.unsqueeze(-1), pred_actions), dim=-1)

        # get the target values
        c_target1 = self.critic.critic(net=self.critic.target_net1, c_state=states, actions=pred_actions,
                                       idx_c=None, idx_a=None, no_absolute_idx=True)
        c_target1 = c_target1.view(batch_size, sample_length)
        c_target2 = self.critic.critic(net=self.critic.target_net2, c_state=states, actions=pred_actions,
                                       idx_c=None, idx_a=None, no_absolute_idx=True)
        c_target2 = c_target2.view(batch_size, sample_length)

        return c_target1, c_target2

    def entropy_loss(self, params_mean, params_L):
        params_mean = params_mean.detach()
        entropy = self.policy.entropy([params_mean, params_L]).mean()
        if self.use_automatic_entropy_tuning:
            alpha = torch.nn.functional.softplus(self.entropy_penalty_coef)
        else:
            alpha = self.entropy_penalty_coef
        entropy_loss = -alpha * entropy
        stats_dict = {"entropy": entropy.item()}
        return entropy_loss, stats_dict

    def _soft_copy(self, target_model, source_model, eta: float = None) -> None:
        """
        This function is to implement the soft copy algorithm
        Args:
            target_model: The target model to be updated
            source_model: The source model to copy from
            eta: percentage of the source model to be used

        Returns:
            None
        """
        eta = self.critic.eta if eta is None else eta
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(eta * source_param.data + (1 - eta) * target_param.data)

    def _check_sampler_dataset(self, dataset, large_value_threshold: float = 1.0e5) -> bool:
        """
        Checks each specified field in the dataset for NaN values, infinite values, and
        abnormally large values. Returns True if all checks are passed.

        Args:
        dataset (dict): The dataset to check.
        large_value_threshold (int): The threshold above which a value is considered too large.

        Returns:
        bool: True if all checks pass, False otherwise.
        """
        for key in ["states", "actions", "rewards", "dones"]:
            data = dataset[key]
            if torch.isnan(data).any():
                print(f"{key} contains NaN")
                return False
            elif torch.isinf(data).any():
                print(f"{key} contains Inf")
                return False
            elif (data.abs() > large_value_threshold).any():
                print(f"{key} contains large values")
                return False
            else:
                continue
                
        # check if the rewards is equal to -50 (QACC invalid, as in fancy_gym setting)
        if self.sampler.reward_scaling is not None and self.sampler.reward_scaling != 1.0:
            if (dataset["rewards"] == -50.0 * self.sampler.reward_scaling).any():
                print("rewards contains QACC invalid")
                return False
        else:
            if (dataset["rewards"] == -50.0).any():
                print("rewards contains QACC invalid")
                return False
        return True