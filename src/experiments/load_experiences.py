import numpy as np


data_dir = "data/experiences"

data = np.load(f"{data_dir}/experiences_transpose-8_1000-episodes.npz")
#data = np.load(f"{data_dir}/experiences_no-permutation_10-episodes.npz")

idx = 1

print(f"State: {data['states'][idx][14]}") # wrong permutation?
print(data["actions"][idx])
print(data["action_masks"][idx]) # wrong permutation?
print(data["dones"][idx])
print(data["episodes"][idx])

print(f"Shape of states: {data['states'].shape}")
print(f"Shape of actions: {data['actions'].shape}")
print(f"Shape of action_masks: {data['action_masks'].shape}")
print(f"Shape of dones: {data['dones'].shape}")

