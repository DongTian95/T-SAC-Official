import os
import sys
import torch
import numpy as np

import random
import multiprocessing
from collections import namedtuple

import mprl.util as util
from . import AbstractAgent
import mprl.rl.critic.n_return_sac_critic as nrsac_critic
import mprl.rl.policy.n_return_sac_policy as nrsac_policy
import mprl.rl.sampler.n_return_sac_sampler as nrsac_sampler
import mprl.rl.replay_buffer.n_return_sac_rb as nrsac_rb


class NReturnSAC(AbstractAgent):
    def __init__(
            self,
            policy: nrsac_policy.NReturnSACPolicy,
            critic: nrsac_critic.NReturnSACCritic,
            sampler: nrsac_sampler.NReturnSACSampler,
            replay_buffer: nrsac_rb.NReturnSACReplayBuffer,
            projection=None,
            dtype=torch.float32,
            device=torch.device("cpu"),
            **kwargs,
    ):
        super().__init__(policy, critic, sampler, dtype=dtype, device=device, **kwargs)

        # for Adam optimizer
        self.betas = kwargs.get("betas")
        if isinstance(self.betas, str):  # deal with betas being a string
            self.betas = tuple(map(float, self.betas.strip('()').split(',')))

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
        self.target_entropy = kwargs.get("target_entropy", "default")
        if self.target_entropy == "default":
            self.target_entropy = -np.prod(self.sampler.debug_env.envs[0].action_space.shape)

        # Optimizers initialization
        self.policy_optimizer = self.get_policy_optimizer(self.policy)
        self.critic_optimizer = self.get_critic_optimizer(self.critic)

        self.temperature_optimizer = torch.optim.AdamW([self.entropy_penalty_coef], lr=self.lr_policy)

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
        
        self.reset_critic_network = kwargs.get("reset_critic_network", None)
        self.reset_policy_network = kwargs.get("reset_policy_network", None)
        self.continue_policy_training = kwargs.get("continue_policy_training", None)

        self.stop_policy_update = None

        self.fresh_agent = True     # is False if loaded from saving file
        self.log_now = False

        # separate policy and entropy training
        self.separate_policy_entropy_training = kwargs.get("separate_policy_entropy_training", False)
        if self.separate_policy_entropy_training:
            self.entropy_optimizer = self.get_entropy_optimizer(self.policy)

        self.update_way = kwargs.get("update_way", "mp")

        self.segments_num = kwargs.get("segments_num", 1)

        self.policy_delay = kwargs.get("policy_delay", 0)
        self.policy_update_index = int(self.policy_delay)

    def get_entropy_optimizer(self, policy):
        variance_net_params = policy.variance_net.parameters()
        entropy_optimizer = torch.optim.AdamW(params=variance_net_params, lr=self.lr_policy,
                                              weight_decay=self.wd_policy, betas=self.betas)
        return entropy_optimizer

    def get_policy_optimizer(self, policy):
        # policy optimizer
        self.policy_net_params = policy.parameters
        policy_optimizer = torch.optim.AdamW(params=policy.parameters, weight_decay=self.wd_policy,
                                             lr=self.lr_policy, betas=self.betas)

        return policy_optimizer

    def get_critic_optimizer(self, critic):
        self.critic_net_params = {
            "net1": critic.net1.parameters(),  # Added parentheses
            "net2": critic.net2.parameters()  # Added parentheses
        }

        critic_opt1, critic_opt2 = self.critic.configure_optimizer(
            weight_decay=self.wd_critic,
            learning_rate=self.lr_critic,
            betas=self.betas,
        )

        return [critic_opt1, critic_opt2]

    def step(self):
        # update total step count
        self.num_iterations += 1

        # if logging data in the current step
        self.log_now = self.evaluation_interval == 1 or self.num_iterations % self.evaluation_interval == 1
        update_critic_now = self.num_global_steps >= self.critic_warmup_step
        update_policy_now = self.num_global_steps >= self.policy_warmup_step

        if not self.fresh_agent:
            buffer_is_ready = self.replay_buffer.is_ready()
            self.num_iterations -= 1    # iteration only for collecting data
        else:
            buffer_is_ready = True

        # collect dataset until buffer size is full
        while not self.replay_buffer.is_ready():
            dataset, num_env_interaction = self.sampler.run(training=True, policy=self.policy, critic=self.critic)
            self.num_global_steps += num_env_interaction
            if self._check_sampler_dataset(dataset):
                self.replay_buffer.add(dataset)
            else:
                print("Discard dataset due to Simulation instability, NaN, Inf or large values")


        if update_critic_now and buffer_is_ready:
            # update agent
            util.run_time_test(lock=True, key="update")

            if self.update_way == "mp":
                critic_loss_dict, policy_loss_dict = self.update(update_policy_now=update_policy_now)
            elif self.update_way == "original_sac":
                critic_loss_dict, policy_loss_dict = self.update_original_sac_way(update_policy_now=update_policy_now)
            else:
                raise NotImplementedError
            
            update_time = util.run_time_test(lock=False, key="update")
            
        else:
            critic_loss_dict, policy_loss_dict = {}, {}
            update_time = 0

        # sample dataset
        util.run_time_test(lock=True, key="sampling")
        dataset, num_env_interaction = self.sampler.run(training=True, policy=self.policy, critic=self.critic)

        self.num_global_steps += num_env_interaction
        sampling_time = util.run_time_test(lock=False, key="sampling")

        # process dataset and save to RB
        util.run_time_test(lock=True, key="process_dataset")
        if self._check_sampler_dataset(dataset):
            self.replay_buffer.add(dataset)
        else:
            print("Discard dataset due to NaN, Inf or large values")
        process_dataset_time = util.run_time_test(lock=False, key="process_dataset")

        # log data
        if self.log_now and buffer_is_ready:
            # generate statistics for env rollouts
            dataset_stats = util.generate_many_stats(dataset, "exploration", to_np=True,
                                                     exception_keys=["decision_idx"])

            # prepare result metrics
            result_metrics = {
                **dataset_stats, **critic_loss_dict, **policy_loss_dict,
                "sampling_time": sampling_time,
                "num_global_steps": self.num_global_steps,
                "update_time": update_time,
                "process_dataset_time": process_dataset_time,
                "lr_policy": self.policy_lr_scheduler.get_last_lr()[0] if self.schedule_lr_policy else self.lr_policy,
                "lr_critic": self.critic_lr_scheduler[0].get_last_lr()[0] if self.schedule_lr_critic else self.lr_critic,
            }

            # evalueate agent
            util.run_time_test(lock=True, key="evaluation")
            evaluate_metrics = util.generate_many_stats(
                self.evaluate()[0], "evaluation", to_np=True,
                exception_keys=["decision_idx"]
            )
            evaluation_time = util.run_time_test(lock=False, key="evaluation")
            result_metrics.update(evaluate_metrics),
            result_metrics.update({"evaluation_time": evaluation_time})
        else:
            result_metrics = {}

        return result_metrics


    def update(self, update_policy_now=False):
        ########################################################################
        #                             Update critic
        ########################################################################
        for _ in range(self.training_times):
            util.run_time_test(lock=True, key="update critic")
            critic_dataset = self.replay_buffer.sample()
            # critic_dataset = self.replay_buffer.prioritized_sample()
            critic_loss_dict = self.update_critic(critic_dataset)   # seems only the last loss dict is used
            update_critic_time = util.run_time_test(lock=False, key="update critic")

            ########################################################################
            #                             Update policy
            ########################################################################

            if update_policy_now:
                util.run_time_test(lock=True, key="update policy")
                # policy_dataset = self.replay_buffer.sample()
                policy_loss_dict = self.update_policy(critic_dataset)
                update_policy_time = util.run_time_test(lock=False, key="update policy")
        if self.schedule_lr_critic:
            self.critic_lr_scheduler[0].step()
            self.critic_lr_scheduler[1].step()
        if self.schedule_lr_policy and update_policy_now:
            self.policy_lr_scheduler.step()
        ########################################################################
        #                             Build log dict
        ########################################################################
        if self.log_now:
            # get critic update statistics
            critic_info_dict = {
                **critic_loss_dict,
                "update_critic_time": update_critic_time,
                "lr_critic": self.critic_lr_scheduler[0].get_last_lr()[0] if self.schedule_lr_critic else self.lr_critic,
            }

            # get policy update statistics
            if update_policy_now:
                policy_info_dict = {
                    **policy_loss_dict,
                    "update_policy_time": update_policy_time,
                    "lr_policy": self.policy_lr_scheduler.get_last_lr() if self.schedule_lr_policy else self.lr_policy,
                }
            else:
                policy_info_dict = {}

        else:
            critic_info_dict = {}
            policy_info_dict = {}

        return critic_info_dict, policy_info_dict

    def update_critic(self, dataset: dict):
        # data extraction
        states = dataset["states"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        dones = dataset["dones"]
        masks = dataset["masks"]

        # get important parameters
        batch_size, sample_length, _ = actions.shape
        step_length = self.sampler.step_length
        min_length = self.sampler.min_length
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
        mask = torch.all(dones.view(batch_size, -1) == 1, dim=0)
        # mask = torch.any(dones.view(batch_size, -1) == 0, dim=0)
        where_result = torch.where(mask)[0]
        if len(where_result) > 0:
            longest_length = where_result[0].item() + 1
        else:
            longest_length = sample_length

        # initialize the loop
        if self.critic_train_in_order == "segment":
            random_indices = self.partition_trajectory_random_lengths_dict_disordered(
                n=longest_length, min_len=min_length, max_len=step_length,
                num_segments=self.random_critic_index_number)
        elif self.critic_train_in_order:  # better for sparse
            random_indices = range(longest_length - 1, -1, -1)
        else:  # better for dense
            num_indices = longest_length if longest_length < self.random_critic_index_number \
                else self.random_critic_index_number
            random_indices = random.sample(range(longest_length), num_indices)

        # critic_loss
        # TODO: this part is horrible, need to be refactored, DON NOT mix list and dict
        for start_idx in random_indices:
            if self.critic_train_in_order == "segment" or self.critic_train_in_order == "segment_uniform":
                end_idx = random_indices[start_idx]
                rand_int = end_idx - start_idx
            else:
                rand_int = torch.randint(1, step_length + 1, (1,)).item()  # to determine the start index
                # rand_int = step_length
                end_idx = start_idx + rand_int
                # update from sample_length to longest_length, TODO: seems no influence
                if end_idx > longest_length:  # reached the beginning of the trajectory
                    end_idx = longest_length
                    rand_int = end_idx - start_idx

            # calculate the target values (policy net doesn't change during one critic update)
            target_states = states[..., end_idx - 1 : end_idx + 1, :]   # this line is to make it consistent to tsac
            c_target1, c_target2 = self.get_target_values(target_states, start_idx=end_idx)

            # get the start state infos
            c_state = states[..., start_idx, :]
            c_actions = actions[..., start_idx, :]
            c_rewards = rewards[..., start_idx: end_idx, :].view(batch_size, -1)
            c_dones = dones[..., end_idx - 1, :]
            c_masks = masks[..., start_idx: end_idx, :].view(batch_size, -1)

            if self.add_timestamp == "only_action":
                # add time stamp to c_actions
                c_actions_time_stamp = util.to_ts(start_idx, self.dtype, self.device)
                c_actions_time_stamp = c_actions_time_stamp.repeat(batch_size, 1)
                c_actions = torch.cat((c_actions_time_stamp, c_state), dim=-1)
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
                c_actions_time_stamp = util.to_ts(start_idx, self.dtype, self.device)
                c_actions_time_stamp = c_actions_time_stamp.repeat(batch_size, 1)
                c_actions = torch.cat((c_actions_time_stamp, c_actions), dim=-1)

            ######################### get the target value #########################


            # whether using random or min to get the target
            if self.random_target:
                # randomly choose the Q value from Q1 or Q2
                target_mask = torch.randint(0, 2, c_target1.shape, device=self.device)
                c_target = target_mask * c_target1 + (1 - target_mask) * c_target2
            else:
                c_target = torch.min(c_target1, c_target2)

            # considering the discount factor
            discounts_factors = torch.tensor([self.discount_factor ** rand_int],
                                             device=self.device, dtype=self.dtype)
            discounts_factors = discounts_factors.repeat(batch_size, 1)
            c_target = c_target * discounts_factors

            # considering the dones or masks
            if self.use_done_for_target:
                c_target = c_target * (1 - c_dones.view(batch_size, -1))
            else:
                c_target = c_target * c_masks[..., -1]

            #####################for report#######################
            # get Monte-Car;p return
            mc_returns = torch.sum(rewards[..., end_idx:, :], dim=1)
            mc_returns = mc_returns.view(batch_size, -1)

            # get the target Monte_Carlo bias
            mc_bias = c_target[..., -1] - mc_returns

            # save the infos
            target_value = util.to_np(c_target.mean())
            target_values_list.append(target_value)

            mc_returns = util.to_np(mc_returns.mean())
            mc_returns_list.append(mc_returns)

            mc_bias = util.to_np(mc_bias.mean())
            target_mc_bias_list.append(mc_bias)

            ###################for report part end################

            # add the discounted reward
            discounted_rewards = torch.zeros_like(c_rewards)
            discounted_rewards[..., 0] = c_rewards[..., 0]
            for idx in range(1, rand_int):
                discounted_rewards[..., idx] = (discounted_rewards[..., idx - 1] +
                                                   self.discount_factor ** idx * c_rewards[..., idx])

            c_target += discounted_rewards
            c_target = c_target.detach()  # just for sure

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

            # get the current values
            c_current1 = self.critic.critic(net=self.critic.net1, c_state=c_state, action=c_actions).view(batch_size, -1)
            c_current2 = self.critic.critic(net=self.critic.net2, c_state=c_state, action=c_actions).view(batch_size, -1)

            # take the mask
            c_target = c_target * c_masks

            c_current1 = c_current1 * c_masks
            c_current2 = c_current2 * c_masks

            # compute the loss
            c_loss1 = torch.nn.functional.mse_loss(c_current1, c_target)
            c_loss2 = torch.nn.functional.mse_loss(c_current2, c_target)

            # the difference of the outputs of critic net shouldn't be significant
            if self.punish_deviation is not None and c_current1.shape[-1] > 1:
                c_loss1 += self.punish_deviation * c_current1.std(dim=1).mean()
                c_loss2 += self.punish_deviation * c_current2.std(dim=1).mean()

            # backpropagation of the critic loss
            # self.scaler.scale(c_loss1).backward()
            # self.scaler.scale(c_loss2).backward()
            c_loss1.backward()
            c_loss2.backward()

            # clip the gradient
            grad_norm_1, grad_norm_c_1 = util.grad_norm_clip(bound=self.clip_grad_norm,
                                                             params=self.critic_net_params["net1"])
            grad_norm_2, grad_norm_c_2 = util.grad_norm_clip(bound=self.clip_critic,
                                                             params=self.critic_net_params["net2"])

            # store the gradient norms and loss
            critic_grad_norm.append(grad_norm_1)
            clipped_critic_grad_norm.append(grad_norm_c_1)
            critic_grad_norm.append(grad_norm_2)
            clipped_critic_grad_norm.append(grad_norm_c_2)

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
            self._soft_copy(target_model=self.critic.target_net1, source_model=self.critic.net1)
            self._soft_copy(target_model=self.critic.target_net2, source_model=self.critic.net2)

        return {
            **util.generate_stats(critic_loss_raw, name="critic_loss"),
            **util.generate_stats(critic_grad_norm, name="critic_grad_norm"),
            **util.generate_stats(clipped_critic_grad_norm, name="clipped_critic_grad_norm"),
            **util.generate_stats(target_values_list, name="target_values"),
            **util.generate_stats(mc_returns_list, name="mc_returns"),
            **util.generate_stats(target_mc_bias_list, name="target_mc_bias"),
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

        # get the longest trajectory
        mask = torch.all(dones.view(batch_size, -1) == 1, dim=0)
        # mask = torch.any(dones.view(batch_size, -1) == 0, dim=0)
        where_result = torch.where(mask)[0]
        if len(where_result) > 0:
            longest_length = where_result[0].item() + 1
        else:
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
            self.policy_optimizer.zero_grad(set_to_none=True)

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

            # generate the params_mean and params_std
            params_mean, params_L = self.policy.policy(obs=c_state)

            # sample the action
            actions = self.policy.sample(params_mean=params_mean, params_L=params_L, require_grad=True)

            # update the entropy penalty coefficient
            if self.use_automatic_entropy_tuning:
                log_probs = self.policy.log_prob(smp_params=actions, params_mean=params_mean, params_L=params_L)
                alpha = torch.nn.functional.softplus(self.entropy_penalty_coef)    # ensures alpha is always positive
                alpha_loss = -(alpha * (log_probs + self.target_entropy).detach()).mean()

                self.temperature_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.temperature_optimizer.step()

            actions = torch.tanh(actions) * action_range
            # actions = actions.unsqueeze(1)  # make compatible to Transformer policy part

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
            q_1 = self.critic.critic(net=self.critic.net1, c_state=c_state, action=actions)
            q_2 = self.critic.critic(net=self.critic.net2, c_state=c_state, action=actions)
            q_value = torch.min(q_1, q_2)
            # q_value = q_1

            # get the surrogate loss
            surrogate_loss = -q_value

            # get the policy loss
            if self.separate_policy_entropy_training:
                policy_loss = surrogate_loss
            else:
                policy_loss = surrogate_loss + entropy_loss

            # take the mask into account
            policy_loss = policy_loss * c_masks.view(batch_size)

            policy_loss = policy_loss.mean()

            # backward
            # self.scaler.scale(policy_loss).backward()
            policy_loss.backward()

            # clip the gradient
            grad_norm, grad_norm_c = util.grad_norm_clip(bound=self.clip_grad_norm,
                                                         params=self.policy_net_params)

            # logging
            surrogate_loss_raw.append(surrogate_loss.mean().item())
            entropy_loss_raw.append(entropy_loss.item())
            policy_loss_raw.append(policy_loss.item())
            entropy.append(entropy_stats["entropy"])
            policy_grad_norm.append(grad_norm)
            clipped_policy_grad_norm.append(grad_norm_c)

            # update the policy
            # self.scaler.step(self.policy_optimizer)
            self.policy_optimizer.step()
            
            if self.separate_policy_entropy_training:
                params_mean, params_L = self.policy.policy(obs=c_state)

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

    @staticmethod
    def partition_trajectory_random_lengths_dict_disordered(n,
                                                            max_len,
                                                            min_len=1,
                                                            num_segments=50):
        """
        Partition indices [0 .. n) into exactly 'num_segments' contiguous segments,
        each length drawn initially from a uniform distribution in [min_len, max_len].
        Then 'correct' the total sum to match n.
        """
        # 1) Feasibility check
        if n < num_segments * min_len or n > num_segments * max_len:
            raise ValueError(
                f"Cannot partition {n} steps into {num_segments} segments "
                f"with lengths in [{min_len}, {max_len}]."
            )

        # 2) Draw each length uniformly from [min_len, max_len]
        lengths = [random.randint(min_len, max_len) for _ in range(num_segments)]
        total = sum(lengths)
        delta = total - n  # can be positive (over) or negative (under)

        # 3) Correct the sum by incrementing/decrementing segments
        #    until sum(lengths) == n
        while delta != 0:
            # pick a random segment index
            i = random.randint(0, num_segments - 1)

            # if sum is too big (delta > 0), try to reduce a segment
            if delta > 0 and lengths[i] > min_len:
                lengths[i] -= 1
                delta -= 1

            # if sum is too small (delta < 0), try to increase a segment
            elif delta < 0 and lengths[i] < max_len:
                lengths[i] += 1
                delta += 1

        # 4) Build the (start, end) pairs
        segments_list = []
        start = 0
        for length in lengths:
            end = start + length
            segments_list.append((start, end))
            start = end

        # 5) Shuffle the segment list for a disordered dictionary
        random.shuffle(segments_list)
        segments_dict = {s: e for (s, e) in segments_list}
        return segments_dict

    @torch.no_grad()
    def get_target_values(self, states, start_idx=1):
        # the first state doesn't need target value
        states = states[:, 1:, :].clone()

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
        params_mean, params_L = self.policy.policy(obs=states)

        # sample the action
        pred_actions = self.policy.sample(params_mean=params_mean, params_L=params_L,
                                          use_mean=False, require_grad=False)
        pred_actions = torch.tanh(pred_actions) * self.sampler.action_range
        # pred_actions = pred_actions.unsqueeze(1)

        if self.add_timestamp == "only_state":
            pass
        elif self.add_timestamp:
            # add time stamp to pred_actions
            indices_tensor = torch.arange(start_idx, sample_length + start_idx, dtype=self.dtype, device=self.device)
            indices_tensor = indices_tensor.repeat(batch_size, 1)
            indices_tensor = indices_tensor.view(batch_size * sample_length, -1)
            pred_actions = torch.cat((indices_tensor.unsqueeze(-1), pred_actions), dim=-1)

        # get the target values
        c_target1 = self.critic.critic(net=self.critic.target_net1, c_state=states, action=pred_actions)
        c_target1 = c_target1.view(batch_size, -1)
        c_target2 = self.critic.critic(net=self.critic.target_net2, c_state=states, action=pred_actions)
        c_target2 = c_target2.view(batch_size, -1)

        return c_target1, c_target2
   
    def entropy_loss(self, params_mean, params_L):
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
        for key in ["states", "actions", "rewards"]:
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
        return True

    def update_original_sac_way(self, update_policy_now=False):
        ########################################################################
        #                             Update critic
        ########################################################################
        for _ in range(self.training_times):
            dataset = self.replay_buffer.sample()
            # critic_dataset = self.replay_buffer.prioritized_sample()
            dones = dataset["dones"]
            batch_size = dones.shape[0]
            mask = torch.all(dones.view(batch_size, -1) == 1, dim=0)
            # mask = torch.any(dones.view(batch_size, -1) == 0, dim=0)
            longest_length = torch.where(mask)[0][0].item() + 1
            segments_indices = self.partition_trajectory_random_lengths_dict_disordered(
                n=longest_length, max_len=self.sampler.step_length, min_len=self.sampler.min_length,
                num_segments=self.segments_num
            )
            for start_idx in segments_indices:
                end_idx= segments_indices[start_idx]
                util.run_time_test(lock=True, key="update critic")
                critic_dataset = {
                    "states": dataset["states"][:, start_idx:end_idx+1, :],
                    "actions": dataset["actions"][:, start_idx:end_idx, :],
                    "rewards": dataset["rewards"][:, start_idx:end_idx, :],
                    "dones": dataset["dones"][:, start_idx:end_idx, :],
                    "masks": dataset["masks"][:, start_idx:end_idx, :],
                }
                critic_loss_dict = self.update_critic(critic_dataset)  # seems only the last loss dict is used
                update_critic_time = util.run_time_test(lock=False, key="update critic")

                ########################################################################
                #                             Update policy
                ########################################################################

                if update_policy_now:
                    if self.policy_delay != 0:
                        self.policy_update_index -= 1
                        if self.policy_update_index != 0:
                            continue
                        else:
                            self.policy_update_index = int(self.policy_delay)
                    util.run_time_test(lock=True, key="update policy")
                    # policy_dataset = self.replay_buffer.sample()
                    # policy_loss_dict = self.update_policy(policy_dataset)
                    policy_loss_dict = self.update_policy(critic_dataset)
                    update_policy_time = util.run_time_test(lock=False, key="update policy")
        if self.schedule_lr_critic:
            self.critic_lr_scheduler[0].step()
            self.critic_lr_scheduler[1].step()
        if self.schedule_lr_policy and update_policy_now:
            self.policy_lr_scheduler.step()
        ########################################################################
        #                             Build log dict
        ########################################################################
        if self.log_now:
            # get critic update statistics
            critic_info_dict = {
                **critic_loss_dict,
                "update_critic_time": update_critic_time,
                "lr_critic": self.critic_lr_scheduler[0].get_last_lr()[0] if self.schedule_lr_critic else self.lr_critic,
            }

            # get policy update statistics
            if update_policy_now:
                policy_info_dict = {
                    **policy_loss_dict,
                    "update_policy_time": update_policy_time,
                    "lr_policy": self.policy_lr_scheduler.get_last_lr() if self.schedule_lr_policy else self.lr_policy,
                }
            else:
                policy_info_dict = {}

        else:
            critic_info_dict = {}
            policy_info_dict = {}

        return critic_info_dict, policy_info_dict