import numpy as np


data_dir = "data/experiences"

data = np.load(f"{data_dir}/experiences_random_1000_episodes.npz")

#print(data["states"][0])
#print(data["actions"][0])
print(data["action_masks"])
print(data["dones"])

print(f"Shape of states: {data['states'].shape}")
print(f"Shape of actions: {data['actions'].shape}")
print(f"Shape of action_masks: {data['action_masks'].shape}")
print(f"Shape of dones: {data['dones'].shape}")
