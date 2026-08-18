import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),

            nn.Linear(256, 256),
            nn.Tanh(),

            nn.Linear(256, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),

            nn.Linear(256, 256),
            nn.Tanh(),

            nn.Linear(256, 1),
        )

        self.log_std = nn.Parameter(
            torch.zeros(action_dim)
        )

    def get_action(self, obs):
        mean = self.actor(obs)

        std = torch.exp(
            self.log_std
        ).expand_as(mean)

        distribution = Normal(
            mean,
            std,
        )

        action = distribution.sample()

        log_prob = distribution.log_prob(
            action
        ).sum(-1)

        value = self.critic(obs).squeeze(-1)

        return action, log_prob, value

    def evaluate_actions(
        self,
        obs,
        actions,
    ):
        mean = self.actor(obs)

        std = torch.exp(
            self.log_std
        ).expand_as(mean)

        distribution = Normal(
            mean,
            std,
        )

        log_prob = distribution.log_prob(
            actions
        ).sum(-1)

        entropy = distribution.entropy().sum(-1)

        value = self.critic(obs).squeeze(-1)

        return (
            log_prob,
            entropy,
            value,
        )


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
        entropy_coef=0.01,
        epochs=10,
        batch_size=64,
    ):
        self.device = torch.device(device)

        self.network = ActorCritic(
            obs_dim,
            action_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=lr,
        )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef

        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.epochs = epochs
        self.batch_size = batch_size

    def select_action(self, obs):
        obs = torch.tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, value = (
                self.network.get_action(obs)
            )

        return (
            action.squeeze(0).cpu().numpy(),
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
        advantages = []

        gae = 0.0

        for t in reversed(
            range(len(rewards))
        ):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            non_terminal = 1.0 - dones[t]

            delta = (
                rewards[t]
                + self.gamma
                * next_val
                * non_terminal
                - values[t]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * non_terminal
                * gae
            )

            advantages.insert(
                0,
                gae,
            )

        returns = [
            adv + val
            for adv, val
            in zip(
                advantages,
                values,
            )
        ]

        return (
            torch.tensor(
                advantages,
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(
                returns,
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def update(
        self,
        observations,
        actions,
        old_log_probs,
        advantages,
        returns,
    ):
        observations = torch.tensor(
            observations,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.float32,
            device=self.device,
        )

        old_log_probs = torch.tensor(
            old_log_probs,
            dtype=torch.float32,
            device=self.device,
        )

        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )

        dataset_size = len(observations)

        for _ in range(self.epochs):
            indices = torch.randperm(
                dataset_size
            )

            for start in range(
                0,
                dataset_size,
                self.batch_size,
            ):
                end = (
                    start
                    + self.batch_size
                )

                batch_indices = (
                    indices[start:end]
                )

                new_log_prob, entropy, value = (
                    self.network.evaluate_actions(
                        observations[
                            batch_indices
                        ],
                        actions[
                            batch_indices
                        ],
                    )
                )

                ratio = torch.exp(
                    new_log_prob
                    - old_log_probs[
                        batch_indices
                    ]
                )

                batch_adv = advantages[
                    batch_indices
                ]

                unclipped = (
                    ratio
                    * batch_adv
                )

                clipped = (
                    torch.clamp(
                        ratio,
                        1 - self.clip_coef,
                        1 + self.clip_coef,
                    )
                    * batch_adv
                )

                policy_loss = -torch.min(
                    unclipped,
                    clipped,
                ).mean()

                value_loss = (
                    (
                        returns[
                            batch_indices
                        ]
                        - value
                    )
                    ** 2
                ).mean()

                entropy_loss = (
                    entropy.mean()
                )

                loss = (
                    policy_loss
                    + self.value_coef
                    * value_loss
                    - self.entropy_coef
                    * entropy_loss
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