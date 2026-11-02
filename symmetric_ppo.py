"""PPO with a left/right mirror symmetry loss (Yu et al., 2018, "Learning Symmetric and
Low-Energy Locomotion"). The policy's mean action on a state should equal the mirrored mean
action on the mirrored state; reward terms alone kept leaving lopsided gaits."""
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F


class SymmetricPPO(PPO):
    """Set `mirror_maps` (obs_perm, obs_sign, act_perm, act_sign from G1Env.mirror_maps())
    and `sym_coef` after construction or loading. Requires a VecNormalize env: stored
    observations are normalized, so they are un-normalized, mirrored and re-normalized."""

    mirror_maps = None
    sym_coef = 0.0

    def _mirror_obs(self, obs):
        obs_perm, obs_sign, _, _ = self.mirror_maps
        rms = self.env.obs_rms
        mean = th.as_tensor(rms.mean, dtype=obs.dtype, device=obs.device)
        std = th.as_tensor(np.sqrt(rms.var + self.env.epsilon), dtype=obs.dtype, device=obs.device)
        raw = obs * std + mean
        mirrored = raw[:, obs_perm] * th.as_tensor(obs_sign, dtype=obs.dtype, device=obs.device)
        return th.clamp((mirrored - mean) / std, -self.env.clip_obs, self.env.clip_obs)

    def _symmetry_loss(self, obs):
        _, _, act_perm, act_sign = self.mirror_maps
        mean_action = self.policy.get_distribution(obs).distribution.mean
        mirrored_mean = self.policy.get_distribution(self._mirror_obs(obs)).distribution.mean
        target = mirrored_mean[:, act_perm] * th.as_tensor(act_sign, dtype=obs.dtype, device=obs.device)
        return F.mse_loss(mean_action, target)

    def train(self) -> None:
        # Copied from stable_baselines3 2.9 PPO.train; the only change is the symmetry loss.
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions, zero_grads = [], []
        sym_losses = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                # Samples whose policy gradient is zero: ratio clipped on the side the advantage pushes it.
                zero_grads.append(th.mean(((advantages > 0) & (ratio > 1 + clip_range)
                                           | (advantages < 0) & (ratio < 1 - clip_range)).float()).item())

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if self.sym_coef > 0 and self.mirror_maps is not None:
                    sym_loss = self._symmetry_loss(rollout_data.observations)
                    sym_losses.append(sym_loss.item())
                    loss = loss + self.sym_coef * sym_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        if sym_losses:
            self.logger.record("train/symmetry_loss", np.mean(sym_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/zero_grad_fraction", np.mean(zero_grads))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
