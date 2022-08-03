# Stable baselines applied to custom OpenAI job-shop environment

Reinforcement Learning applied to permutation job shop problems


## Requirements

- Stable-baselines3: https://github.com/DLR-RM/stable-baselines3
- OpenAI gym: 
- OpenAI gym box2d:
- Swig (requirement to install box2d?)
- An OpenAi Gym environment for the Job Shop Scheduling problem: https://github.com/prosysscience/JSSEnv 



## Tutorials for stable-baselines3

- [x] [Reinforcement Learning with Stable Baselines 3 - Introduction (P.1)](https://youtu.be/XbWhJdQgi7E)
- [ ] [Saving and Loading Models - Stable Baselines 3 Tutorial (P.2)](https://youtu.be/dLP-2Y6yu70)
- [ ] [Custom Environments - Reinforcement Learning with Stable Baselines 3 (P.3)](https://youtu.be/uKnjGn8fF70)
- [ ] [Tweaking Custom Environment Rewards - Reinforcement Learning with Stable Baselines 3 (P.4)](https://youtu.be/yvwxbkKX9dc)

## Possible extensions

- https://github.com/DLR-RM/rl-baselines3-zoo

## Action masking

- Maskable PPO: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
- Paper "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms": https://arxiv.org/abs/2006.14171
- Blog post: https://costa.sh/blog-a-closer-look-at-invalid-action-masking-in-policy-gradient-algorithms.html
- Another blog post: https://boring-guy.sh/posts/masking-rl/