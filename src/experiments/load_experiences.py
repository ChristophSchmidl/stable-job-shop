import numpy as np


data_dir = "data/experiences/test"

#data = np.load(f"{data_dir}/experiences_random_100-episodes.npz")
#data = np.load(f"{data_dir}/experiences_transpose-8_1000-episodes.npz")
data = np.load(f"{data_dir}/experiences_no-permutation_1000-episodes.npz")

idx = 0

print(f"State: {data['states'][idx]}") # wrong permutation?
print(data["actions"][idx])
print(data["action_masks"][idx]) # wrong permutation?
print(data["dones"][idx])
print(data["episodes"][idx])

print(f"Shape of states: {data['states'].shape}")
print(f"Shape of actions: {data['actions'].shape}")
print(f"Shape of action_masks: {data['action_masks'].shape}")
print(f"Shape of dones: {data['dones'].shape}")

