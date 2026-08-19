import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


def init_layer(layer, std=1.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.actor = nn.Sequential(
            init_layer(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            init_layer(nn.Linear(256, 256)),
            nn.Tanh(),
            init_layer(nn.Linear(256, action_dim), 0.01),
        )

        self.critic = nn.Sequential(
            init_layer(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            init_layer(nn.Linear(256, 256)),
            nn.Tanh(),
            init_layer(nn.Linear(256, 1)),
        )

        self.log_std = nn.Parameter(
            torch.ones(action_dim) * -0.5
        )

    def _distribution(self, obs):
        mean = self.actor(obs)

        log_std = torch.clamp(
            self.log_std,
            -5.0,
            1.0,
        )

        std = torch.exp(log_std).expand_as(mean)

        return Normal(mean, std)

    def get_action(self, obs):
        dist = self._distribution(obs)

        raw_action = dist.rsample()

        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action).sum(-1)

        log_prob -= torch.log(
            1.0 - action.pow(2) + 1e-6
        ).sum(-1)

        value = self.critic(obs).squeeze(-1)

        return action, raw_action, log_prob, value

    def evaluate_actions(self, obs, raw_actions):
        dist = self._distribution(obs)

        actions = torch.tanh(raw_actions)

        log_prob = dist.log_prob(raw_actions).sum(-1)

        log_prob -= torch.log(
            1.0 - actions.pow(2) + 1e-6
        ).sum(-1)

        entropy = dist.entropy().sum(-1)

        value = self.critic(obs).squeeze(-1)

        return log_prob, entropy, value


class PPO:
    def __init__(
        self,
        obs_dim,
        action_dim,
        device="cpu",
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        value_coef=0.5,
        entropy_coef=0.005,
        epochs=10,
        batch_size=128,
    ):
        self.device = torch.device(device)

        self.network = ActorCritic(
            obs_dim,
            action_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=lr,
            eps=1e-5,
        )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size

    def select_action(self, obs):
        obs = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            action, raw_action, log_prob, value = (
                self.network.get_action(obs)
            )

        return (
            action.squeeze(0).cpu().numpy(),
            raw_action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    def compute_gae(
        self,
        rewards,
        values,
        dones,
        next_value,
    ):
        advantages = torch.zeros(
            len(rewards),
            dtype=torch.float32,
            device=self.device,
        )

        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            non_terminal = 1.0 - dones[t]

            delta = (
                rewards[t]
                + self.gamma * next_val * non_terminal
                - values[t]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * non_terminal
                * gae
            )

            advantages[t] = gae

        values_tensor = torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        )

        returns = advantages + values_tensor

        return advantages, returns

    def update(
        self,
        observations,
        raw_actions,
        old_log_probs,
        advantages,
        returns,
    ):
        observations = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=self.device,
        )

        raw_actions = torch.as_tensor(
            raw_actions,
            dtype=torch.float32,
            device=self.device,
        )

        old_log_probs = torch.as_tensor(
            old_log_probs,
            dtype=torch.float32,
            device=self.device,
        )

        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )

        dataset_size = observations.shape[0]

        for _ in range(self.epochs):
            indices = torch.randperm(
                dataset_size,
                device=self.device,
            )

            for start in range(
                0,
                dataset_size,
                self.batch_size,
            ):
                batch = indices[
                    start:start + self.batch_size
                ]

                new_log_prob, entropy, value = (
                    self.network.evaluate_actions(
                        observations[batch],
                        raw_actions[batch],
                    )
                )

                ratio = torch.exp(
                    new_log_prob
                    - old_log_probs[batch]
                )

                batch_adv = advantages[batch]

                loss_1 = ratio * batch_adv

                loss_2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_coef,
                    1.0 + self.clip_coef,
                ) * batch_adv

                policy_loss = -torch.min(
                    loss_1,
                    loss_2,
                ).mean()

                value_loss = (
                    returns[batch] - value
                ).pow(2).mean()

                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    0.5,
                )

                self.optimizer.step()

    def save(self, path):
        torch.save(
            self.network.state_dict(),
            path,
        )

    def load(self, path):
        self.network.load_state_dict(
            torch.load(
                path,
                map_location=self.device,
            )
        )