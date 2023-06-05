import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from gym import spaces

# fully connected neural network with one hidden layer
class SimpleFFNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, dropout_value, checkpoint_dir):
        super().__init__()
        self.dropout_value = dropout_value
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # https://stackoverflow.com/questions/50376463/
        # pytorch-whats-the-difference-between-define-layer-in-init-and-directly-us

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dims, 256)
        self.fc2 = nn.Linear(256, 128)
        self.drop = nn.Dropout(dropout_value)
        self.fc3 = nn.Linear(128, n_actions)

        '''
        self.base = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dims, 128), 
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, n_actions) 
        )
        '''

        # loss and optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr) # Adam optimizer
        self.loss = nn.CrossEntropyLoss() # combines nn.LogSoftmax() and nn.NLLLoss() in one single class
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        ''' Forward pass through the network'''
        state = state.float()
        state = self.flatten(state)
        state = self.fc1(state)
        state = F.relu(state)
        state = self.drop(state)
        state = self.fc2(state)
        state = F.relu(state)
        state = self.drop(state)
        actions = self.fc3(state)

        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')

        if os.path.isfile(self.checkpoint_file):
            print("Loading model from file: ", self.checkpoint_file)
            # map_location is required to ensure that a model that is trained on GPU can be run on CPU
            self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))
        else:
            print(f"File not found: {self.checkpoint_file}. Continue training from scratch.")

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        self.eval()
        observation = T.from_numpy(observation['real_obs'])
        #print(observation)
        
        original_outputs = self.forward(observation)

        outputs = original_outputs.flatten().detach().numpy()
        action_masks = action_masks.flatten()
        outputs[~action_masks] = -np.inf # Invalid actions are set to -infinity
        #print(outputs)

        #_, actions = T.max(outputs, 0)
        actions = np.argmax(outputs)
        #print(action_masks)
        #print(actions)
        #print(actions)

        return [actions], []