import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

class STDP_QLearning:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table = np.zeros((state_size, action_size))
        
        self.tau_plus = 20.0
        self.tau_minus = 20.0
        self.A_plus = 0.1
        self.A_minus = 0.12
        
        self.pre_spikes = [[] for _ in range(state_size)]
        self.post_spikes = [[] for _ in range(action_size)]
        
        self.memory = deque(maxlen=2000)
        
    def stdp_update(self, pre_neuron, post_neuron, current_time):
        delta_w = 0
        
        for t_pre in self.pre_spikes[pre_neuron]:
            if t_pre < current_time:
                delta_t = t_pre - current_time
                delta_w -= self.A_minus * np.exp(delta_t / self.tau_minus)
        
        for t_post in self.post_spikes[post_neuron]:
            if t_post < current_time:
                delta_t = current_time - t_post
                delta_w += self.A_plus * np.exp(-delta_t / self.tau_plus)
        
        return delta_w
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done, time_step):
        self.pre_spikes[state].append(time_step)
        
        q_target = reward
        if not done:
            q_target += self.gamma * np.max(self.q_table[next_state])
        
        self.post_spikes[action].append(time_step)
        
        stdp_factor = self.stdp_update(state, action, time_step)
        
        error = q_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * error + stdp_factor
        
        self.cleanup_spikes(time_step)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def cleanup_spikes(self, current_time, max_history=100):
        for i in range(self.state_size):
            self.pre_spikes[i] = [t for t in self.pre_spikes[i] 
                                 if current_time - t < max_history]
        for i in range(self.action_size):
            self.post_spikes[i] = [t for t in self.post_spikes[i] 
                                  if current_time - t < max_history]
    
    def replay(self, batch_size, current_time):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            self.learn(state, action, reward, next_state, done, current_time)


class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.state = 0
        self.goal = size * size - 1
        self.obstacles = [12, 17, 18]
        
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self, action):
        row, col = self.state // self.size, self.state % self.size
        
        if action == 0 and row > 0:
            row -= 1
        elif action == 1 and col < self.size - 1:
            col += 1
        elif action == 2 and row < self.size - 1:
            row += 1
        elif action == 3 and col > 0:
            col -= 1
        
        new_state = row * self.size + col
        
        if new_state == self.goal:
            self.state = new_state
            return new_state, 10, True
        elif new_state in self.obstacles:
            return self.state, -5, False
        else:
            self.state = new_state
            return new_state, -0.1, False


if __name__ == "__main__":
    env = GridWorld(size=5)
    state_size = env.size * env.size
    action_size = 4
    agent = STDP_QLearning(state_size, action_size)
    
    episodes = 1000
    batch_size = 32
    
    rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        time_step = 0
        
        while not done and time_step < 100:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            agent.learn(state, action, reward, next_state, done, time_step)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            time_step += 1
            
        # Experience replay
        agent.replay(min(batch_size, len(agent.memory)), time_step)
            
        rewards.append(total_reward)
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('stdp_qlearning_rewards.png')
    plt.show()
    
    print("Final Q-table:")
    for i in range(env.size):
        for j in range(env.size):
            state = i * env.size + j
            best_action = np.argmax(agent.q_table[state])
            action_symbol = ["↑", "→", "↓", "←"][best_action]
            print(f"{action_symbol} ", end="")
        print()