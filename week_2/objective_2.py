# Implementing a basic Q-learning agent

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import gym

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon      # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Exploration
        else:
            return np.argmax(self.q_table[state])  # Exploitation
    
    def update(self, state, action, reward, next_state, done):
        # Q-learning update rule: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        
        # Decay exploration rate
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(env, agent, episodes=1000):
    rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = tuple(state) if isinstance(state, np.ndarray) else state
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = tuple(next_state) if isinstance(next_state, np.ndarray) else next_state
            
            agent.update(state, action, reward, next_state, done or truncated)
            state = next_state
            total_reward += reward
            
            if done or truncated:
                break
                
        rewards.append(total_reward)
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
    return rewards

def plot_rewards(rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('q_learning_rewards.png')
    plt.show()

# Example usage with FrozenLake environment
if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', is_slippery=False)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    agent = QLearningAgent(state_size, action_size)
    rewards = train_agent(env, agent, episodes=1000)
    plot_rewards(rewards)
    
    # Test the trained agent
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = np.argmax(agent.q_table[state])
        next_state, reward, done, truncated, _ = env.step(action)
        state = next_state
        total_reward += reward
        env.render()
        
        if done or truncated:
            break
            
    print(f"Test reward: {total_reward}")