import numpy as np
import torch
from torch.utils.data import random_split
import pytest
import random
import logging
from IPython.display import display
from sympy.combinatorics import Permutation
from sympy.interactive import init_printing
from src.utils.permutation_handler import PermutationHandler
from src.supervised_learning.dataset import Ta41Dataset


@pytest.fixture(scope="module")
def dataset():
    """
    dataset fixture - returns a dataset
    """
    try:
        dataset = Ta41Dataset.get_normal_dataset() # custom class
        logging.info("Created normal Ta41Dataset: SUCCESS")
    except FileNotFoundError as err:
        logging.error("dataset fixture: The file wasn't found.")
        raise err

    return dataset

def test_random_splits(dataset):
    train, test, valid = random_split(dataset, [0.8, 0.1, 0.1])
    len_dataset = len(dataset)
    len_train = len(train)
    len_test = len(test)
    len_valid = len(valid)
    len_sum_of_splits = len_train + len_test + len_valid

    print(f"Length of full dataset: {len_dataset}")
    print(f"Length of train: {len_train}")
    print(f"Length of test: {len_test}")
    print(f"Length of valid: {len_valid}")
    print(f"Summed lengths of splits: {len_sum_of_splits}")