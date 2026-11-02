"""DISC: PPO with dimension-wise importance-sampling weight clipping (Han & Sung, ICML 2019,
"Dimension-Wise Importance Sampling Weight Clipping for Sample-Efficient Reinforcement Learning").
Follows the authors' code (github.com/seungyulhan/disc, baselines/DISC/DISC.py):

* Each action dimension's IS weight is clipped separately, so a sample keeps its gradient
  unless every dimension is clipped. PPO clips the joint ratio, which with 23 dimensions
  drifts far from 1 and zeroes the gradient of most samples.
* An adaptive IS loss (alpha_IS * 0.5 * mean(log rho)^2 on the current batch) bounds the update.
* The last `replay_length` rollouts are reused if their mean per-dimension IS deviation is
  below `eps_b`, with advantages and value targets from GAE-V (GAE with V-trace traces).

Built on SymmetricPPO, so the mirror symmetry loss applies the same way as in the baseline.
"""
import numpy as np
import torch as th
from torch.distributions import Normal
from torch.nn import functional as F
from stable_baselines3.common.utils import explained_variance
from symmetric_ppo import SymmetricPPO


class DISCPPO(SymmetricPPO):
    """Defaults are the paper's (Table A.1; the authors' run_disc.py uses j_targ 1e-3 instead),
    except replay_length: the paper keeps 64 batches of 2048 samples; 16 rollouts of
    8 envs x 1024 steps hold the same 131k samples."""

    def __init__(self, *args, disc_eps=0.4, j_targ=1e-4, eps_b=0.1, replay_length=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.disc_eps = disc_eps
        self.j_targ = j_targ
        self.eps_b = eps_b
        self.replay_length = replay_length
        self.alpha_is = 1.0
        self.replay = []

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["replay"]

    def _norm_stats(self):
        venv = self.get_vec_normalize_env()
        return (venv.obs_rms.mean, np.sqrt(venv.obs_rms.var + venv.epsilon),
                float(np.sqrt(venv.ret_rms.var + venv.epsilon)), venv.clip_obs, venv.clip_reward)

    def _dist(self, obs):
        return self.policy.get_distribution(obs).distribution

    @th.no_grad()
    def _forward_chunks(self, obs, actions, chunk=32768):
        """Per-dimension log-probs of `actions` and values, for a large array of normalized obs."""
        logp, values = [], []
        for i in range(0, len(obs), chunk):
            o = th.as_tensor(obs[i:i + chunk], device=self.device)
            a = th.as_tensor(actions[i:i + chunk], device=self.device)
            logp.append(self._dist(o).log_prob(a).cpu().numpy())
            values.append(self.policy.predict_values(o).flatten().cpu().numpy())
        return np.concatenate(logp), np.concatenate(values)

    def _store_rollout(self):
        """Add the rollout just collected to the replay buffer. Observations and rewards are kept
        un-normalized and re-normalized with the current VecNormalize stats at each update."""
        buf = self.rollout_buffer
        mean, std, rscale, _, _ = self._norm_stats()
        T, E = buf.observations.shape[:2]
        flat_obs = buf.observations.reshape(T * E, -1)
        flat_act = buf.actions.reshape(T * E, -1)
        beh_logp, _ = self._forward_chunks(flat_obs, flat_act)
        self.replay.append({
            "obs": (buf.observations * std + mean).astype(np.float32),
            "rew": (buf.rewards * rscale).astype(np.float32),  # includes SB3's timeout bootstrap
            "starts": buf.episode_starts.copy(),
            "act": buf.actions.copy(),
            "beh_logp": beh_logp.reshape(T, E, -1),
        })
        self.replay = self.replay[-self.replay_length:]
        # Rollouts are contiguous per env, so only the newest one's final obs is needed.
        self._final_obs = self._last_obs * std + mean
        self._final_start = self._last_episode_starts.astype(np.float32)

    def _gae_v(self, rew, values, last_value, starts, last_start, rho):
        """GAE-V (eq. 7): A_t = delta_t + gamma*lambda*min(1, rho_{t+1})*A_{t+1}, target
        V_t + min(1, rho_t)*A_t. Arrays are (time, env); reduces to GAE when rho = 1."""
        c = np.minimum(1.0, rho)
        adv = np.zeros_like(rew)
        last = np.zeros(rew.shape[1], dtype=np.float32)
        for t in reversed(range(len(rew))):
            if t == len(rew) - 1:
                nonterminal, next_value = 1.0 - last_start, last_value
            else:
                nonterminal, next_value = 1.0 - starts[t + 1], values[t + 1]
            delta = rew[t] + self.gamma * next_value * nonterminal - values[t]
            adv[t] = delta + self.gamma * self.gae_lambda * nonterminal * last
            last = c[t] * adv[t]
        return adv, values + c * adv

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        self._store_rollout()

        # Re-evaluate the whole buffer under the current policy and value function.
        mean, std, rscale, clip_obs, clip_rew = self._norm_stats()
        K = len(self.replay)
        T, E, D = self.replay[0]["act"].shape
        obs = np.clip((np.concatenate([b["obs"] for b in self.replay]) - mean) / std,
                      -clip_obs, clip_obs).astype(np.float32)
        act = np.concatenate([b["act"] for b in self.replay])
        beh_logp = np.concatenate([b["beh_logp"] for b in self.replay])
        rew = np.clip(np.concatenate([b["rew"] for b in self.replay]) / rscale, -clip_rew, clip_rew)
        starts = np.concatenate([b["starts"] for b in self.replay])
        logp, values = self._forward_chunks(obs.reshape(K * T * E, -1), act.reshape(K * T * E, D))
        logp, values = logp.reshape(K * T, E, D), values.reshape(K * T, E)
        final_obs = np.clip((self._final_obs - mean) / std, -clip_obs, clip_obs).astype(np.float32)
        with th.no_grad():
            last_value = self.policy.predict_values(th.as_tensor(final_obs, device=self.device)).flatten().cpu().numpy()
        log_rho_dim = logp - beh_logp
        rho = np.exp(np.minimum(log_rho_dim.sum(-1), 20.0))
        adv, returns = self._gae_v(rew, values, last_value, starts, self._final_start, rho)

        # Batch inclusion (eq. 5): mean over samples and dims of |1 - rho_d| < eps_b. The current
        # rollout (last in the buffer) is always used.
        dev_dim = np.abs(np.exp(log_rho_dim) - 1.0).reshape(K, T * E * D).mean(axis=1)
        included = [k for k in range(K - 1) if dev_dim[k] < self.eps_b] + [K - 1]

        def flat(x, k):
            return x[k * T:(k + 1) * T].reshape(T * E, *x.shape[2:])

        to_t = lambda x: th.as_tensor(np.ascontiguousarray(x), dtype=th.float32, device=self.device)
        data = {name: to_t(np.concatenate([flat(x, k) for k in included]))
                for name, x in (("obs", obs), ("act", act), ("beh_logp", beh_logp), ("adv", adv),
                                ("ret", returns), ("rho", rho))}
        n_on = T * E
        n_off = len(data["obs"]) - n_on
        on_idx = np.arange(n_off, n_off + n_on)

        # As in the authors' code: a fixed number of gradient steps per epoch; each minibatch has
        # batch_size on-policy samples plus a proportional share of the reused samples.
        steps_per_epoch = n_on // self.batch_size
        mb_off = (len(included) - 1) * self.batch_size
        eps = self.disc_eps
        pg_losses, value_losses, entropy_losses, is_losses, sym_losses = [], [], [], [], []
        zero_grad, any_clipped, approx_kl = [], [], []
        for epoch in range(self.n_epochs):
            for _ in range(steps_per_epoch):
                idx = np.random.choice(on_idx, self.batch_size)
                if mb_off > 0:
                    idx = np.concatenate([np.random.randint(0, n_off, mb_off), idx])
                idx = th.as_tensor(idx, device=self.device)
                mb = {name: x[idx] for name, x in data.items()}
                on_w = th.zeros(len(idx), device=self.device)
                on_w[-self.batch_size:] = len(included)  # mean(x * on_w) = mean over on-policy samples

                dist = self._dist(mb["obs"])
                values_pred = self.policy.predict_values(mb["obs"]).flatten()
                log_ratio_dim = th.clamp(dist.log_prob(mb["act"]) - mb["beh_logp"], -20.0, 20.0)
                ratio_dim = th.exp(log_ratio_dim)

                # Advantages normalized so that mean(min(1, rho) * A) = 0 (authors' code).
                c = th.clamp(mb["rho"], max=1.0)
                radv = c * mb["adv"]
                advantages = (mb["adv"] - radv.mean() / c.mean()) / (radv.std() + 1e-8)

                # Per dimension: min(rho_d, clip(rho_d)) = min(rho_d, 1 + eps) for A > 0 and
                # max(rho_d, clip(rho_d)) = max(rho_d, 1 - eps) for A < 0; then the product over
                # dimensions, computed in log space.
                pos = (advantages > 0).unsqueeze(1)
                log_factor = th.where(pos, th.clamp(log_ratio_dim, max=float(np.log(1 + eps))),
                                      th.clamp(log_ratio_dim, min=float(np.log(1 - eps))))
                r = th.exp(log_factor.sum(-1))
                policy_loss = -(r * advantages / r.mean().detach()).mean()

                log_ratio = log_ratio_dim.sum(-1)
                is_loss = 0.5 * (log_ratio ** 2 * on_w).mean()
                value_loss = F.mse_loss(mb["ret"], values_pred)
                entropy_loss = -dist.entropy().sum(-1).mean()
                loss = (policy_loss + self.alpha_is * is_loss + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss)
                if self.sym_coef > 0 and self.mirror_maps is not None:
                    sym_loss = self._symmetry_loss(mb["obs"])
                    sym_losses.append(sym_loss.item())
                    loss = loss + self.sym_coef * sym_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                with th.no_grad():
                    # A dimension's gradient vanishes when it is clipped on the side the advantage
                    # pushes it; a sample's vanishes when all its dimensions are.
                    dim_dead = th.where(pos, ratio_dim > 1 + eps, ratio_dim < 1 - eps)
                    on = slice(-self.batch_size, None)
                    zero_grad.append(dim_dead[on].all(-1).float().mean().item())
                    any_clipped.append(dim_dead[on].any(-1).float().mean().item())
                    lr_on = log_ratio[on]
                    approx_kl.append(th.mean(th.exp(lr_on) - 1 - lr_on).item())
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                is_losses.append(is_loss.item())
            self._n_updates += 1

        # Adapt alpha_IS (eq. 3), bounded as in the authors' code.
        is_mean = float(np.mean(is_losses))
        if is_mean > self.j_targ * 1.5:
            self.alpha_is *= 2.0
        elif is_mean < self.j_targ / 1.5:
            self.alpha_is /= 2.0
        self.alpha_is = float(np.clip(self.alpha_is, 2 ** -10, 64))

        cur = slice((K - 1) * T, K * T)
        explained_var = explained_variance(values[cur].flatten(), returns[cur].flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        if sym_losses:
            self.logger.record("train/symmetry_loss", np.mean(sym_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl))
        self.logger.record("train/clip_fraction", np.mean(any_clipped))
        self.logger.record("train/zero_grad_fraction", np.mean(zero_grad))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("disc/is_loss", is_mean)
        self.logger.record("disc/alpha_is", self.alpha_is)
        self.logger.record("disc/batches_used", len(included))
        self.logger.record("disc/buffer_batches", K)
        self.logger.record("disc/rho_dim_dev_oldest", float(dev_dim[0]))
