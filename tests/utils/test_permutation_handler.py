import numpy as np
import torch
import pytest
from src.utils.permutation_handler import PermutationHandler

TEST_DATA_LIST = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36]]
TEST_DATA_NDARRAY = np.array(TEST_DATA_LIST)


@pytest.mark.parametrize("data", [TEST_DATA_LIST, TEST_DATA_NDARRAY])
def test_permute(data):
    
    # This is a random permutation
    random_permuted_data, perm_matrix, perm_indices = PermutationHandler.permute(data)

    assert type(random_permuted_data) is np.ndarray
    assert type(perm_matrix) is torch.Tensor
    assert type(perm_indices) is np.ndarray

    # This is a permutation based on perm_indices. The result should be the same as the random permutation.
    permuted_data, _, _ = PermutationHandler.permute(data, perm_indices)

    assert np.array_equal(random_permuted_data, permuted_data)

@pytest.mark.parametrize("data", [TEST_DATA_LIST, TEST_DATA_NDARRAY])
def test_inverse_permute(data):
    # https://steemit.com/numpy/@luoq/how-to-compute-the-inverse-of-permuation-in-numpy
    # This is a random permutation
    random_permuted_data, perm_matrix, perm_indices = PermutationHandler.permute(data)

    assert type(random_permuted_data) is np.ndarray
    assert type(perm_matrix) is torch.Tensor
    assert type(perm_indices) is np.ndarray

    # Reverting the random permutation should result in the original data
    reverted_data = PermutationHandler.inverse_permute(random_permuted_data, perm_matrix)
    reverted_data = np.asarray(reverted_data)

    assert np.array_equal(reverted_data, data)