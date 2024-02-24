# A Supervised Learning Approach to Robust Reinforcement Learning for Job Shop Scheduling

A three-step-approach that combines RL and Supervised Learning techniques. The initially trained RL policy is used as a labelling oracle that generates state-action pairs which are then augmented with varying permutation percentages to transpose job orders. These state-action pairs serve as data sets for re-training models in a supervised learning setup that uses Dropout layers to improve robustness.


## Usage

- ``python -m src.main --help``

## Requirements

- Stable-baselines3: https://github.com/DLR-RM/stable-baselines3
- OpenAI gym: 
- OpenAI gym box2d:
- Swig (requirement to install box2d?)
- An OpenAi Gym environment for the Job Shop Scheduling problem: https://github.com/prosysscience/JSSEnv 