import copy
import os
import sys
import torch

import random
import multiprocessing

import mprl.util as util
import mprl.rl.critic.transformer_sac_critic as tsac_critic
import mprl.rl.policy.transformer_sac_policy as tsac_policy
import mprl.rl.sampler.transformer_sac_sampler as tsac_sampler
import mprl.rl.replay_buffer.transformer_sac_replay_buffer as tsac_rb
from .transformer_sac import TransformerSAC

class TransformerSACMultiProcessing(TransformerSAC):
    def __init__(
            self,
            policy: tsac_policy.TransformerSACPolicy,
            critic: tsac_critic.TransformerSACCritic,
            sampler: tsac_sampler.TransformerSACSampler,
            conn: multiprocessing.connection.Connection,
            replay_buffer: tsac_rb.TransformerSACReplayBuffer,
            projection=None,
            dtype=torch.float32,
            device=torch.device("cpu"),
            **kwargs,
    ):
        super().__init__(
            policy=policy,
            critic=critic,
            sampler=sampler,
            replay_buffer=replay_buffer,
            projection=projection,
            dtype=dtype,
            device=device,
            **kwargs,
        )
        self.conn = conn

        self.log_now = False
        self.fresh_agent = True     # is False if loaded from saving file

        self.reset_critic_network = kwargs.get("reset_critic_network", None)
        self.reset_policy_network = kwargs.get("reset_policy_network", None)
        self.continue_policy_training = kwargs.get("continue_policy_training", None)

        self.stop_policy_update = None

        self.update_way = kwargs.get("update_way", "mp")

        self.segments_num = kwargs.get("segments_num", 1)

        self.policy_delay = kwargs.get("policy_delay", 0)
        self.policy_update_index = int(self.policy_delay)

    def step(self):
        # update total step count
        self.num_iterations += 1

        # if logging data in the current step
        self.log_now = self.evaluation_interval == 1 or self.num_iterations % self.evaluation_interval == 1

        if not self.fresh_agent:
            buffer_is_ready = self.replay_buffer.is_ready()
            if self.num_iterations > 2:
                self.num_iterations -= 1    # iteration only for collecting data
        else:
            buffer_is_ready = self.replay_buffer.is_ready()

        update_critic_now = self.num_iterations >= self.critic_warmup_step and buffer_is_ready
        update_policy_now = self.num_iterations >= self.policy_warmup_step and buffer_is_ready

        # collect dataset until buffer size is full
        while not self.replay_buffer.is_ready() and self.num_iterations > 1:
            self.conn.send(self.policy.parameters)
            dataset, num_env_interaction = self.conn.recv()
            self.num_global_steps += num_env_interaction
            if self._check_sampler_dataset(dataset):
                self.replay_buffer.add(dataset)
            else:
                print("Discard dataset due to NaN, Inf or large values")

        util.run_time_test(lock=True, key="sampling")

        # NOTE: update parameter of policy in subprocess
        self.conn.send(self.policy.parameters)
        if update_critic_now and buffer_is_ready:
            # update agent
            util.run_time_test(lock=True, key="update")

            # Reset the critic network
            if self.reset_critic_network is not None:
                if self.num_iterations in self.reset_critic_network:
                    self.critic.reset_network()
                    self.critic_optimizer = self.get_critic_optimizer(self.critic)  # after reset the net, optimizer also needs to be resetted
                    self.stop_policy_update = True

            # Reset the policy network
            if self.reset_policy_network is not None:
                if self.num_iterations in self.reset_policy_network:
                    self.policy.reset_network()
                    self.policy_optimizer = self.get_policy_optimizer(self.policy)

            if self.continue_policy_training is not None:
                if self.num_iterations in self.continue_policy_training:
                    self.stop_policy_update = False

            if self.stop_policy_update is not None:
                update_policy_now = not self.stop_policy_update and update_policy_now

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

        # NOTE: wait for data from subprocess of sampling  process
        dataset, num_env_interaction = self.conn.recv()

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
        if self.log_now:
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

            # evaluate agent
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
                policy_dataset = self.replay_buffer.sample()
                policy_loss_dict = self.update_policy(policy_dataset)
                # policy_loss_dict = self.update_policy(critic_dataset)
                if self.use_trust_region_update:
                    if self.use_last_three_as_policy_target:
                        if self.num_iterations == 1:
                            self.last_policy = copy.deepcopy(self.policy)
                        elif self.num_iterations == 2:
                            self.second_last_policy = copy.deepcopy(self.last_policy)
                            self.last_policy = copy.deepcopy(self.policy)
                        elif self.num_iterations >= 3:
                            self.third_last_policy = copy.deepcopy(self.second_last_policy)
                            self.second_last_policy = copy.deepcopy(self.last_policy)
                            self.last_policy = copy.deepcopy(self.policy)
                        self.old_policy = self.use_several_net_as_target_net([
                            self.last_policy, self.second_last_policy, self.third_last_policy
                        ], self.old_policy)
                    else:
                        # use soft copy to update old policy
                        self._soft_copy(self.old_policy.mean_net, self.policy.mean_net,
                                        eta=self.policy_polyak_average_factor)
                        self._soft_copy(self.old_policy.variance_net, self.policy.variance_net,
                                        eta=self.policy_polyak_average_factor)
                # policy_loss_dict = self.update_policy(critic_dataset)
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

    def update_original_sac_way(self, update_policy_now=False):
        ########################################################################
        #                             Update critic and policy
        ########################################################################
        for _ in range(self.training_times):
            # critic_dataset = self.replay_buffer.sample()
            update_dataset = self.replay_buffer.prioritized_sample()
            critic_loss_dict, policy_loss_dict = self.update_original_sac(update_dataset,
                                                                          update_policy_now=update_policy_now)
            ########################################################################
            #                             Update TRPL related
            ########################################################################

            if update_policy_now:
                if self.use_trust_region_update:
                    if self.use_last_three_as_policy_target:
                        if self.num_iterations == 1:
                            self.last_policy = copy.deepcopy(self.policy)
                        elif self.num_iterations == 2:
                            self.second_last_policy = copy.deepcopy(self.last_policy)
                            self.last_policy = copy.deepcopy(self.policy)
                        elif self.num_iterations >= 3:
                            self.third_last_policy = copy.deepcopy(self.second_last_policy)
                            self.second_last_policy = copy.deepcopy(self.last_policy)
                            self.last_policy = copy.deepcopy(self.policy)
                        self.old_policy = self.use_several_net_as_target_net([
                            self.last_policy, self.second_last_policy, self.third_last_policy
                        ], self.old_policy)
                    else:
                        # use soft copy to update old policy
                        self._soft_copy(self.old_policy.mean_net, self.policy.mean_net,
                                        eta=self.policy_polyak_average_factor)
                        self._soft_copy(self.old_policy.variance_net, self.policy.variance_net,
                                        eta=self.policy_polyak_average_factor)
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
                "lr_critic": self.critic_lr_scheduler[0].get_last_lr()[0] if self.schedule_lr_critic else self.lr_critic,
            }

            # get policy update statistics
            if update_policy_now:
                policy_info_dict = {
                    **policy_loss_dict,
                    "lr_policy": self.policy_lr_scheduler.get_last_lr() if self.schedule_lr_policy else self.lr_policy,
                }
            else:
                policy_info_dict = {}

        else:
            critic_info_dict = {}
            policy_info_dict = {}

        return critic_info_dict, policy_info_dict


    @staticmethod
    def partition_trajectory_with_optional_discard(n,
                                                   max_len,
                                                   min_len=1,
                                                   num_segments=50,
                                                   discard_allowed=True):
        """
        Partition indices [0 .. n) into exactly 'num_segments' segments, each length in [min_len, max_len].
        If 'discard_allowed' is True, we allow skipping some points when n > num_segments * max_len.
        Otherwise, we raise a ValueError as in the original code.
        """

        # 1) Check if we have "too few" points to fill segments at their min length:
        if n < num_segments * min_len:
            raise ValueError(
                f"Cannot partition {n} steps into {num_segments} segments each at least length {min_len}."
            )

        # 2) If we have more points than the maximum feasible coverage (n > num_segments * max_len),
        #    and if discarding is allowed, reduce n by discarding.
        if n > num_segments * max_len:
            if discard_allowed:
                # Number of points to discard:
                to_discard = n - (num_segments * max_len)
                # Suppose you want to discard them at random from the set of indices [0..n).
                # For large n, it's more memory-efficient to do random sampling.
                all_indices = list(range(n))
                random.shuffle(all_indices)

                # Actually remove them from consideration:
                # (We only keep n - to_discard = num_segments * max_len indices)
                kept_indices = sorted(all_indices[to_discard:])  # keep the last portion
                n = len(kept_indices)  # now n == num_segments * max_len
            else:
                raise ValueError(
                    f"Cannot partition {n} steps into {num_segments} segments with max_len={max_len} unless we discard."
                )

        # 3) Now we know: num_segments * min_len <= n <= num_segments * max_len
        lengths = [random.randint(min_len, max_len) for _ in range(num_segments)]
        total = sum(lengths)
        delta = total - n

        # 4) Correct the sum
        while delta != 0:
            i = random.randint(0, num_segments - 1)
            if delta > 0 and lengths[i] > min_len:
                lengths[i] -= 1
                delta -= 1
            elif delta < 0 and lengths[i] < max_len:
                lengths[i] += 1
                delta += 1

        # 5) Build segments
        segments_list = []
        start = 0
        for length in lengths:
            end = start + length
            segments_list.append((start, end))
            start = end

        # 6) Shuffle the segments for a "disordered" dict
        random.shuffle(segments_list)
        segments_dict = {s: e for (s, e) in segments_list}
        return segments_dict