# soc_synaptirl_23b2260

## Knowledge before participating in this project

- Basic ML algorithms
- No previous knowledge of RL / SNNs

## Week 1

During Week 1 I referred to some introductory lectures and literature on SNNs and RL which helped me familiarize with these new concepts.

## Week 2

### Objective 1
I implemented a Spiking Neural Network (SNN) to classify digits in the MNIST dataset. I focused on binary classification between digits 0 and 1, creating a custom SpikingNeuron class that models membrane potential and spike generation. I built a multi-layer SNN architecture with linear layers followed by spiking neurons, and trained it using standard backpropagation with a custom temporal integration mechanism.

During this process, I learned about the fundamentals of spiking neural networks, membrane potential dynamics, spike generation mechanisms, and how to adapt traditional neural network architectures for spike-based computation. I also gained experience with PyTorch's tensor operations and the MNIST dataset preprocessing pipeline.

### Objective 2
I developed a Q-learning agent to solve the FrozenLake environment from the Gymnasium library. I implemented an epsilon-greedy exploration strategy with decay to balance exploration and exploitation. I modified the reward structure to encourage more efficient path-finding, and visualized the learning progress with reward plots. The agent successfully learned to navigate through the grid to reach the goal while avoiding holes.

During this process, I learned about the core principles of reinforcement learning, including the Q-learning algorithm, exploration-exploitation tradeoff, reward shaping techniques, and environment modeling. I also gained experience with the Gymnasium framework, policy evaluation, and visualization of learning performance.

### Objective 3
I created a hybrid STDP-QLearning algorithm for a custom GridWorld environment. I implemented Spike-Timing-Dependent Plasticity (STDP) to model synaptic weight updates based on the relative timing of pre and post-synaptic neuron spikes. I combined this with traditional Q-learning and experience replay to improve learning efficiency. The agent learned to navigate around obstacles to reach the goal state, and I visualized the optimal policy using directional arrows.

During this process, I learned about biologically-inspired learning mechanisms, particularly STDP as a model of synaptic plasticity, and how to integrate it with traditional reinforcement learning approaches. I also gained experience with experience replay buffers, custom environment design, and policy visualization techniques.

## Week 3

### Objective 1
I designed a more complex spiking neural network model to solve a dynamic maze environment. I created a GridMaze class with randomly generated obstacles and implemented a full SNN agent with multiple spiking neurons connected by STDP synapses. I used reward-modulated STDP to adjust synaptic weights based on both spike timing and reward signals. I implemented functions to visualize the training process with real-time maze rendering and performance metrics, showing how the agent gradually learned to find the shortest path to the goal.

During this process, I learned about advanced neuromorphic computing concepts, including reward-modulated plasticity, multi-layer spiking networks, and dynamic environment interactions. I also gained experience with real-time visualization of neural network training, batch processing techniques, and implementing complex reinforcement learning algorithms with biologically-inspired mechanisms.

## Week 4

### Objective 1
I explored the field of meta-learning and its applications in neuromorphic computing. I learned about meta-learning as a paradigm where systems learn how to learn, enabling rapid adaptation to new tasks with minimal data. I studied few-shot learning techniques that allow models to generalize from very few examples, which is crucial for resource-constrained environments. I investigated cross-domain few-shot learning in mobile OCR, understanding how transfer learning principles can be applied to recognize text in various contexts and visual conditions with limited training samples. I also delved into Hebbian meta-learning, discovering how biologically-inspired Hebbian plasticity rules can be incorporated into meta-learning frameworks to create more adaptive and efficient learning systems.

During this process, I gained insights into how meta-learning bridges the gap between traditional deep learning and more biologically plausible learning mechanisms. I developed an understanding of how these approaches can significantly reduce data requirements while maintaining performance, particularly in edge computing scenarios. I also learned how to implement and evaluate different meta-learning algorithms, comparing their efficiency and adaptability across various problem domains.

### Objective 2
I implemented a Model-Agnostic Meta-Learning (MAML) approach for maze navigation tasks. I designed a flexible MazeEnvironment class capable of generating random maze variations to support meta-learning across different maze configurations. I built a neural network policy that learns to generalize navigation strategies across multiple maze layouts, implementing both inner-loop adaptation for specific mazes and outer-loop meta-updates to improve general maze-solving capabilities.

During this process, I learned about the two-level optimization process in MAML, with fast adaptation to new tasks during inner loops and slower meta-optimization across task distributions in outer loops. I gained experience in implementing policy gradient methods with returns calculation and advantage normalization techniques. I developed skills in creating controlled environment variations to test transfer learning capabilities, and I learned how to measure and compare pre-adaptation and post-adaptation performance to quantify the benefits of meta-learning. I also practiced visualizing agent behavior and learning progress in spatially structured reinforcement learning tasks.

### Objective 3
I implemented a standard Q-learning agent to solve maze navigation tasks and compared its performance with the MAML approach. I created a comprehensive maze environment with random maze generation capabilities and developed a Q-learning agent with state discretization techniques to handle the continuous state space. I conducted extensive comparative experiments, evaluating both approaches across multiple randomly generated maze configurations and visualizing the learning process through performance metrics and maze rendering.

During this process, I learned about the fundamental differences between meta-learning and traditional reinforcement learning approaches, particularly in terms of adaptation speed and transfer learning capabilities. I gained experience with state representation techniques for reinforcement learning, implementing discretization methods for continuous state spaces, and epsilon-greedy exploration strategies with decay mechanisms. I developed skills in comprehensive performance evaluation, creating visualizations of learning curves for rewards, success rates, and steps per episode, and conducting controlled experiments to compare algorithm performance across different problem instances.

## Week 5

### Objective 1
I learned about policy gradient methods in reinforcement learning, which are a family of algorithms that directly optimize the policy function mapping states to actions. I implemented the REINFORCE algorithm, also known as Monte Carlo policy gradient, which uses complete episode returns to update the policy parameters. I designed a neural network policy for the CartPole environment and trained it to balance the pole by directly maximizing expected rewards.

During this process, I learned about the fundamental principles of policy-based reinforcement learning, which optimizes the policy directly rather than learning a value function like in Q-learning. I gained experience implementing stochastic policies using probability distributions over actions, which enables better exploration of the state space. I learned how to compute policy gradients using the log-likelihood ratio trick and how to reduce gradient variance through baseline subtraction and advantage normalization techniques. I also developed skills in implementing neural network policies with PyTorch, using categorical distributions for discrete action spaces, and visualizing the learning progress through reward curves.

## Assignment

For my final assignment, I implemented a complete REINFORCE algorithm from scratch to solve the CartPole-v1 environment. I created a PyTorch-based policy network architecture with a fully connected neural network that outputs action probabilities through a softmax layer. The implementation features a stochastic policy that samples actions from a categorical distribution, allowing the agent to explore the environment effectively.

I implemented key components of the policy gradient algorithm including:
1. Trajectory collection with stochastic action selection
2. Return calculation with discounted rewards
3. Advantage normalization to reduce variance
4. Policy gradient computation using the log-probability ratio trick
5. Neural network weight updates via backpropagation

During the training process, I monitored the agent's learning progress by tracking episode rewards and visualizing the learning curve. The agent successfully learned to balance the pole on the cart, achieving high rewards within a relatively small number of training episodes. I also implemented an evaluation function that uses the trained policy in a deterministic manner (selecting the highest probability action) to demonstrate the final performance.

This assignment allowed me to apply my understanding of policy gradient methods to a concrete reinforcement learning problem, reinforcing the theoretical concepts learned throughout the project with practical implementation experience.

