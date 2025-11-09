import numpy as np
from typing import Dict, List, Tuple, Union, Callable, Optional
from collections import defaultdict
import random

class MDP:
    """
    A class representing a finite Markov Decision Process (MDP) as defined by the tuple (S, A, mu0, T, r, gamma, H).
    Assumes discrete, finite state and action spaces for tabular representation.
    """
    
    def __init__(self, S: List[str], A: List[str], mu0: Dict[str, float], 
                 T: Dict[Tuple[str, str], Dict[str, float]], 
                 r: Dict[Tuple[str, str, str], float], 
                 gamma: float = 1.0, H: Union[int, float] = float('inf')):
        """ 
        Args:
            S (List[str]): List of state identifiers.
            A (List[str]): List of action identifiers.
            mu0 (Dict[str, float]): Initial state distribution {state: probability}.
            T (Dict[Tuple[str, str], Dict[str, float]]): Transition dynamics {(s, a): {s': probability}}.
            r (Dict[Tuple[str, str, str], float]): Reward function {(s, a, s'): reward}.
            gamma (float): Discount factor in [0, 1].
            H (Union[int, float]): Horizon; int for finite, float('inf') for infinite.
        """
        self.S = S
        self.A = A
        self.mu0 = mu0
        self.T = T
        self.r = r
        self.gamma = gamma
        self.H = H if H != float('inf') else np.inf
        
        # assert abs(sum(mu0.values()) - 1.0) < 1e-6, "mu0 must be a valid probability distribution."
        # for s in S:
        #     assert s in mu0, f"State {s} missing from mu0."
        #     for a in A:
        #         assert (s, a) in T, f"Transition missing for ({s}, {a})."
        #         trans = T[(s, a)]
        #         assert abs(sum(trans.values()) - 1.0) < 1e-6, f"T({s}, {a}) must be a valid distribution."
        #         for s_prime in trans:
        #             assert s_prime in S, f"Invalid next state {s_prime} in T({s}, {a})."
        #         assert (s, a, s_prime) in r for s_prime in trans, f"Reward missing for transitions from ({s}, {a})."
    
    def sample_initial_state(self) -> str:
        return random.choices(list(self.mu0.keys()), weights=list(self.mu0.values()))[0]
    
    def sample_transition(self, s: str, a: str) -> Tuple[str, float]:
        trans = self.T[(s, a)]
        s_prime = random.choices(list(trans.keys()), weights=list(trans.values()))[0]
        reward = self.r[(s, a, s_prime)]
        return s_prime, reward
    
    def step(self, s: str, a: str) -> Tuple[str, float]:
        return self.sample_transition(s, a)
    
    def is_terminal(self, t: int, s: str) -> bool:
        """Checks if the episode should terminate (for finite horizon)."""
        if self.H < np.inf:
            return t >= self.H
        return False  # Infinite horizon; no terminal states assumed here
    
    def simulate_episode(self, policy: Callable[[str], Dict[str, float]]) -> Tuple[List[Tuple[str, str]], List[float], float]:
        """
        Args:
            policy (Callable[[str], Dict[str, float]]): Policy π(s) -> {a: probability}.
        
        Returns:
            Tuple[List[Tuple[str, str]], List[float], float]: (trajectory of (s_t, a_t), rewards [r_0, ..., r_{H-1}], total discounted return).
        """
        trajectory = []
        rewards = []
        s = self.sample_initial_state()
        t = 0
        total_return = 0.0
        discount = 1.0
        
        while not self.is_terminal(t, s):
            # Sample action
            action_dist = policy(s)
            a = random.choices(list(action_dist.keys()), weights=list(action_dist.values()))[0]
            
            # Step
            s_next, r = self.step(s, a)
            
            # Accumulate
            trajectory.append((s, a))
            rewards.append(r)
            total_return += discount * r
            
            # Update
            s = s_next
            discount *= self.gamma
            t += 1
        
        return trajectory, rewards, total_return
    
    def expected_return(self, policy: Callable[[str], Dict[str, float]], num_episodes: int = 1000) -> float:
        """
        Estimates the expected return η(π) via Monte Carlo sampling.

        Args:
            policy (Callable[[str], Dict[str, float]]): Policy to evaluate.
            num_episodes (int): Number of episodes to simulate.
        
        Returns:
            float: Estimated average discounted return.
        """
        returns = []
        for _ in range(num_episodes):
            _, _, return_val = self.simulate_episode(policy)
            returns.append(return_val)
        return np.mean(returns)

# simple policy optimizer (value iteration for deterministic policies)
def create_deterministic_policy_from_values(V: Dict[str, float], mdp: MDP) -> Callable[[str], Dict[str, float]]:
    """Creates a deterministic policy from value function V via greedy selection."""
    def policy(s: str) -> Dict[str, float]:
        q_values = {}
        for a in mdp.A:
            expected_q = sum(mdp.T[(s, a)][s_next] * (mdp.r[(s, a, s_next)] + mdp.gamma * V.get(s_next, 0)) 
                             for s_next in mdp.T[(s, a)])
            q_values[a] = expected_q
        best_a = max(q_values, key=q_values.get)
        return {best_a: 1.0}
    return policy

def value_iteration(mdp: MDP, tol: float = 1e-6, max_iter: int = 1000) -> Tuple[Dict[str, float], Callable[[str], Dict[str, float]]]:
    """
    Performs value iteration to find the optimal value function V* and corresponding policy π*.
    Assumes finite horizon H=1 for simplicity; extend for general H if needed.
    For infinite horizon with gamma < 1.
    
    Returns:
        Tuple[Dict[str, float], Callable]: (V*, π*).
    """
    V = {s: 0.0 for s in mdp.S}
    
    for _ in range(max_iter):
        V_new = {}
        delta = 0
        for s in mdp.S:
            max_q = float('-inf')
            for a in mdp.A:
                q = sum(mdp.T[(s, a)][s_next] * (mdp.r[(s, a, s_next)] + mdp.gamma * V[s_next]) 
                        for s_next in mdp.T[(s, a)])
                max_q = max(max_q, q)
            V_new[s] = max_q
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < tol:
            break
    
    pi_star = create_deterministic_policy_from_values(V, mdp)
    return V, pi_star

# Example: Simple GridWorld MDP (4x4 grid, goal at (3,3), actions up/down/left/right)
def create_gridworld_mdp() -> MDP:
    S = [(i, j) for i in range(4) for j in range(4)]
    A = ['up', 'down', 'left', 'right']
    
    # Initial distribution: uniform
    mu0 = {s: 1.0 / len(S) for s in S}
    
    # Transitions (deterministic for simplicity, but add noise if desired)
    T = {}
    r = {}
    for s in S:
        i, j = s
        for a in A:
            if a == 'up':
                s_next = (min(i-1, 0), j) if i > 0 else s
            elif a == 'down':
                s_next = (min(i+1, 3), j) if i < 3 else s
            elif a == 'left':
                s_next = (i, max(j-1, 0)) if j > 0 else s
            else:  # right
                s_next = (i, min(j+1, 3)) if j < 3 else s
            
            # Reward: -1 per step, +10 at goal
            reward = 10.0 if s_next == (3, 3) else -1.0
            
            T[(s, a)] = {s_next: 1.0}
            r[(s, a, s_next)] = reward
    
    mdp = MDP(S, A, mu0, T, r, gamma=0.9, H=np.inf)
    return mdp

# Demonstration
if __name__ == "__main__":
    mdp = create_gridworld_mdp()
    
    # Random policy
    def random_policy(s: str) -> Dict[str, float]:
        return {a: 1.0 / len(mdp.A) for a in mdp.A}
    
    eta_random = mdp.expected_return(random_policy, num_episodes=100)
    print(f"Expected return under random policy: {eta_random:.2f}")
    
    # Optimal policy via value iteration
    V_star, pi_star = value_iteration(mdp)
    eta_star = mdp.expected_return(pi_star, num_episodes=100)
    print(f"Expected return under optimal policy: {eta_star:.2f}")
    
    # Simulate one episode under optimal policy
    traj, rews, ret = mdp.simulate_episode(pi_star)
    print(f"Sample trajectory length: {len(traj)}, return: {ret:.2f}")
    print(f"Trajectory: {traj[:5]}...")  # First 5 steps