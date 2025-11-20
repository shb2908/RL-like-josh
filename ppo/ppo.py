import math
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def act(self, state, memory: Memory):
        if isinstance(state, (list, tuple)):
            state = torch.tensor(state, dtype=torch.float32, device=device)
        elif not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()

        memory.states.append(state.squeeze(0).detach())
        memory.actions.append(action.squeeze(0).detach())
        memory.logprobs.append(dist.log_prob(action).squeeze(0).detach())

        return action.item()

    def evaluate(self, states, actions):
        if states.dim() == 1:
            states = states.unsqueeze(0)

        action_probs = self.actor(states)
        dist = Categorical(action_probs)

        if actions.dim() == 0:
            actions = actions.unsqueeze(0)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(states).squeeze(-1)

        return action_logprobs, state_values, dist_entropy


class PPO:
    """Proximal Policy Optimization (clipped) with GAE and minibatching."""
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_size=64,
        lr=3e-4,
        betas=(0.9, 0.999),
        gamma=0.99,
        K_epochs=4,
        eps_clip=0.2,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        batch_size=64,
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.batch_size = batch_size

        self.policy = ActorCritic(state_dim, action_dim, hidden_size).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.MseLoss = nn.MSELoss()

    def compute_gae(self, rewards, values, is_terminals):
        """Compute GAE advantages and returns given rewards and value estimates.
        rewards, values: 1D tensors (T,)
        is_terminals: list/iterable of booleans (T,)
        Returns: advantages (T,), returns (T,)
        """
        advantages = torch.zeros_like(rewards, device=device)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - float(is_terminals[t])
                nextvalues = 0.0
            else:
                nextnonterminal = 1.0 - float(is_terminals[t])
                nextvalues = values[t + 1]

            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values
        return advantages, returns

    def update(self, memory: Memory):
        old_states = torch.stack(memory.states).to(device)
        old_actions = torch.stack(memory.actions).to(device)
        old_logprobs = torch.stack(memory.logprobs).to(device)

        rewards = torch.tensor(memory.rewards, dtype=torch.float32, device=device)
        is_terminals = memory.is_terminals

        with torch.no_grad():
            _, values, _ = self.policy_old.evaluate(old_states, old_actions)

        advantages, returns = self.compute_gae(rewards, values, is_terminals)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = old_states.size(0)

        for _ in range(self.K_epochs):
            # minibatch sampling
            for indices in BatchSampler(SubsetRandomSampler(range(dataset_size)), self.batch_size, drop_last=False):
                batch_states = old_states[indices]
                batch_actions = old_actions[indices]
                batch_logprobs = old_logprobs[indices]
                batch_returns = returns[indices]
                batch_advantages = advantages[indices]

                # evaluate current policy
                logprobs, state_values, dist_entropy = self.policy.evaluate(batch_states, batch_actions)

                # ratios for clipped surrogate objective
                ratios = torch.exp(logprobs - batch_logprobs.detach())

                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # critic loss (value function)
                critic_loss = self.MseLoss(state_values, batch_returns)

                # entropy bonus
                entropy_loss = -dist_entropy.mean()

                loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

        # copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())