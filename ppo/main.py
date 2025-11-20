import time
import gym
import torch
from ppo.ppo import PPO, Memory


def train(
    env_name="CartPole-v1",
    total_timesteps=200_000,
    timesteps_per_update=2048,
    save_interval=10,
    print_interval=10,
):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo = PPO(state_dim, action_dim, hidden_size=64, lr=3e-4, K_epochs=10, batch_size=64)
    memory = Memory()

    timestep = 0
    update_iter = 0
    episode = 0
    episode_reward = 0
    rewards_history = []

    state = env.reset()
    start_time = time.time()

    while timestep < total_timesteps:
        # collect a rollout of timesteps_per_update steps
        for _ in range(timesteps_per_update):
            action = ppo.policy_old.act(state, memory)
            next_state, reward, done, _ = env.step(action)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            episode_reward += reward
            timestep += 1

            state = next_state

            if done:
                state = env.reset()
                rewards_history.append(episode_reward)
                episode += 1
                if episode % print_interval == 0:
                    avg_reward = sum(rewards_history[-print_interval:]) / print_interval
                    elapsed = time.time() - start_time
                    print(f"Episode {episode}	Timestep {timestep}	AvgReward(last {print_interval}) {avg_reward:.2f}	Elapsed {elapsed:.1f}s")
                episode_reward = 0

            if timestep >= total_timesteps:
                break

        # update policy using collected rollout
        ppo.update(memory)
        memory.clear_memory()
        update_iter += 1

        if update_iter % save_interval == 0:
            torch.save(ppo.policy.state_dict(), f"ppo_{env_name}_policy_{update_iter}.pt")

    # final save
    torch.save(ppo.policy.state_dict(), f"ppo_{env_name}_policy_final.pt")
    env.close()


if __name__ == "__main__":
    train()
