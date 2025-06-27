import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import gymnasium as gym

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.5, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            if np.all(self.q_table[state] == self.q_table[state][0]):
                return random.randint(0, self.action_size - 1)
            return int(np.argmax(self.q_table[state]))
    
    def update(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (1 - int(done))
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def debug_q_table(self):
        print("Q-table non-zero values:")
        for state, values in self.q_table.items():
            if np.any(values != 0):
                print(f"State {state}: {values}")

def train_agent(env, agent, episodes=1000):
    rewards = []
    best_reward = 0
    best_episode = 0
    episode_lengths = []
    action_mapping = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = tuple(state) if isinstance(state, np.ndarray) else state
        total_reward = 0
        done = False
        steps = 0
        max_steps = 100
        episode_actions = []
        
        while not done and steps < max_steps:
            action = agent.get_action(state)
            episode_actions.append(action)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = tuple(next_state) if isinstance(next_state, np.ndarray) else next_state
            
            modified_reward = reward
            if done and reward == 0:
                modified_reward = -1.0
            elif next_state == state and not done:
                modified_reward = -0.1
            
            if episode % 100 == 0 and steps < 5:
                print(f"Step {steps}: State: {state}, Action: {action} ({action_mapping[action]}), Next: {next_state}, Reward: {modified_reward}")
                print(f"Q-values for state {state}: {agent.q_table[state]}")
            
            agent.update(state, action, modified_reward, next_state, done or truncated)
            state = next_state
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
                
        rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode
        
        if episode % 100 == 0:
            avg_length = sum(episode_lengths[-100:]) / min(100, len(episode_lengths))
            print(f"Episode: {episode}, Steps: {steps}, Total Reward: {total_reward}, Best: {best_reward} (ep {best_episode}), Epsilon: {agent.epsilon:.2f}")
            print(f"Avg episode length (last 100): {avg_length:.1f}")
            print(f"Actions taken: {[action_mapping[a] for a in episode_actions]}")
            agent.debug_q_table()
            print("-" * 50)
    
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

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="ansi")
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    print(f"State size: {state_size}, Action size: {action_size}")
    print("FrozenLake Map:")
    env.reset()
    print(env.render())
    
    agent = QLearningAgent(
        state_size=state_size, 
        action_size=action_size, 
        alpha=0.8,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.1
    )
    
    rewards = train_agent(env, agent, episodes=2000)
    plot_rewards(rewards)
    
    print("\n" + "="*50)
    print("TESTING THE TRAINED AGENT")
    print("="*50)
    
    env_test = gym.make('FrozenLake-v1', is_slippery=False, render_mode="ansi")
    state, _ = env_test.reset()
    state = tuple(state) if isinstance(state, np.ndarray) else state
    
    print("\nFull Q-table:")
    action_mapping = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
    for s in range(state_size):
        action_values = [f"{a}:{action_mapping[a]}={v:.2f}" for a, v in enumerate(agent.q_table[s])]
        print(f"State {s}: {action_values}")
    
    print("\nOptimal path according to Q-table:")
    test_state = 0
    path = [test_state]
    steps_taken = 0
    
    state_transitions = {
        0: {0: 0, 1: 4, 2: 1, 3: 0},
        1: {0: 0, 1: 5, 2: 2, 3: 1},
        2: {0: 1, 1: 6, 2: 3, 3: 2},
        3: {0: 2, 1: 7, 2: 3, 3: 3},
        4: {0: 4, 1: 8, 2: 5, 3: 0},
        5: {0: 4, 1: 9, 2: 6, 3: 1},
        6: {0: 5, 1: 10, 2: 7, 3: 2},
        7: {0: 6, 1: 11, 2: 7, 3: 3},
        8: {0: 8, 1: 12, 2: 9, 3: 4},
        9: {0: 8, 1: 13, 2: 10, 3: 5},
        10: {0: 9, 1: 14, 2: 11, 3: 6},
        11: {0: 10, 1: 15, 2: 11, 3: 7},
        12: {0: 12, 1: 12, 2: 13, 3: 8},
        13: {0: 12, 1: 13, 2: 14, 3: 9},
        14: {0: 13, 1: 14, 2: 15, 3: 10},
        15: {0: 14, 1: 15, 2: 15, 3: 11}
    }
    
    while not test_state == 15 and steps_taken < 20:
        best_action = np.argmax(agent.q_table[test_state])
        next_test_state = state_transitions[test_state][best_action]
        print(f"From state {test_state}, take action {best_action} ({action_mapping[best_action]}) to reach state {next_test_state}")
        
        test_state = next_test_state
        path.append(test_state)
        steps_taken += 1
        
        if test_state == 5 or test_state == 7 or test_state == 8 or test_state == 11 or test_state == 12:
            print(f"Fell into a hole at state {test_state}!")
            break
        elif test_state == 15:
            print(f"Reached the goal at state {test_state}!")
            break
    
    print(f"Path taken: {path}")
    
    print("\nRunning the agent in the environment:")
    state, _ = env_test.reset()
    state = tuple(state) if isinstance(state, np.ndarray) else state
    total_reward = 0
    done = False
    steps = 0
    
    print(env_test.render())
    
    while not done and steps < 100:
        action = np.argmax(agent.q_table[state])
        next_state, reward, done, truncated, _ = env_test.step(action)
        next_state = tuple(next_state) if isinstance(next_state, np.ndarray) else next_state
        
        print(f"Step {steps}: State {state}, Action {action} ({action_mapping[action]}), Reward: {reward}")
        print(env_test.render())
        
        state = next_state
        total_reward += reward
        steps += 1
        
        if done or truncated:
            break
            
    print(f"Test reward: {total_reward}, Steps: {steps}")