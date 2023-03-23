import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.utils import get_project_root
import numpy as np
import math
import os


class CustomDataset():
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        return {
            "x": torch.tensor(current_sample, dtype=torch.float),
            "y": torch.tensor(current_target, dtype=torch.float)
        }

class Ta41Dataset(Dataset):

    def __init__(self, file_name="experiences_no-permutation_1000-episodes.npz"):
        # data loading
        self.data_dir = os.path.join(get_project_root(), "data/experiences")  
        self.data = np.load(f"{self.data_dir}/experiences_transpose-8_1000-episodes.npz")
        self.x = torch.from_numpy(self.data["states"])
        self.y = torch.from_numpy(self.data["actions"])
        self.n_samples = self.x.shape[0] # number of samples in the dataset

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.n_samples

    @staticmethod
    def get_normal_dataset():
        return Ta41Dataset(file_name="experiences_no-permutation_1000-episodes.npz")

    @staticmethod
    def get_randomly_permuted_dataset():
        return Ta41Dataset(file_name="experiences_random_1000-episodes.npz")

    @staticmethod
    def get_transposed_dataset(n_swaps=1):
        if n_swaps < 1 or n_swaps > 15:
            raise ValueError("n_swaps must be between 1 and 15")

        return Ta41Dataset(file_name=f"experiences_transpose-{n_swaps}_1000-episodes.npz")

    @staticmethod
    def get_normal_dataloader(batch_size=4, shuffle=True):
        return DataLoader(dataset=Ta41Dataset(file_name="experiences_no-permutation_1000-episodes.npz"), batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def get_randomly_permuted_dataloader(batch_size=4, shuffle=True):
        return DataLoader(dataset=Ta41Dataset(file_name="experiences_random_1000-episodes.npz"), batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def get_transposed_dataloader(n_swaps=1, batch_size=4, shuffle=True):
        if n_swaps < 1 or n_swaps > 15:
            raise ValueError("n_swaps must be between 1 and 15")

        return DataLoader(dataset=Ta41Dataset(file_name=f"experiences_transpose-{n_swaps}_1000-episodes.npz"), batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    dataloader = Ta41Dataset.get_randomly_permuted_dataloader()

    #########################
    #   Using iterator
    #########################
    dataiter = iter(dataloader)
    states, actions = dataiter.next()
    print(f"Shape of states: {states.shape}")
    print(f"Shape of actions: {actions.shape}")
    print(states, actions)

    #########################
    #   Using for loop
    #########################
    for states, actions in dataloader:
        print(f"Shape of states: {states.shape}")
        print(f"Shape of actions: {actions.shape}")
        print(states, actions)
        break

    train_data, test_data, train_targets, test_targets = train_test_split(states, actions, stratify=actions)