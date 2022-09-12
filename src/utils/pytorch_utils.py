import numpy as np
import torch
import random

def enforce_deterministic_behavior():
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

def get_device():
    # Device configuration
    # See also: https://wandb.ai/wandb/common-ml-errors/reports/How-To-Use-GPU-with-PyTorch---VmlldzozMzAxMDk
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def get_device_name():
    device = get_device()
    return torch.cuda.get_device_name(device)

def get_device_count():
    return torch.cuda.device_count()

def get_device_memory():
    device = get_device()
    return torch.cuda.get_device_properties(device).total_memory