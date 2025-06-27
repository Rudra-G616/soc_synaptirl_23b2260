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
