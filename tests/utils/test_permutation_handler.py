import numpy as np
import torch
import pytest
import random
from IPython.display import display
from sympy.combinatorics import Permutation
from sympy.interactive import init_printing
from src.utils.permutation_handler import PermutationHandler

init_printing(perm_cyclic=False, pretty_print=True)

TEST_DATA_LIST_EVEN = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36]]
TEST_DATA_NDARRAY_EVEN = np.array(TEST_DATA_LIST_EVEN)
TEST_DATA_LIST_ODD = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30]]
TEST_DATA_NDARRAY_ODD = np.array(TEST_DATA_LIST_ODD)



@pytest.mark.parametrize("data", [TEST_DATA_LIST_EVEN, TEST_DATA_NDARRAY_EVEN, TEST_DATA_LIST_ODD, TEST_DATA_NDARRAY_ODD])
def test_permute(data):
    
    # This is a random permutation
    random_permuted_data, perm_indices = PermutationHandler.permute(data)

    assert type(random_permuted_data) is np.ndarray
    #assert type(perm_matrix) is torch.Tensor
    assert type(perm_indices) is np.ndarray

    # This is a permutation based on perm_indices. The result should be the same as the random permutation.
    permuted_data, _ = PermutationHandler.permute(data, perm_indices)

    assert np.array_equal(random_permuted_data, permuted_data)

@pytest.mark.parametrize("data", [TEST_DATA_LIST_EVEN, TEST_DATA_NDARRAY_EVEN, TEST_DATA_LIST_ODD, TEST_DATA_NDARRAY_ODD])
def test_inverse_permute(data):
    # https://steemit.com/numpy/@luoq/how-to-compute-the-inverse-of-permuation-in-numpy
    # This is a random permutation
    random_permuted_data, perm_indices = PermutationHandler.permute(data)

    assert type(random_permuted_data) is np.ndarray
    #assert type(perm_matrix) is torch.Tensor
    assert type(perm_indices) is np.ndarray

    # Reverting the random permutation should result in the original data
    reverted_data = PermutationHandler.inverse_permute(random_permuted_data, perm_indices)

    assert np.array_equal(reverted_data, data)

@pytest.mark.parametrize("data", [TEST_DATA_LIST_EVEN, TEST_DATA_NDARRAY_EVEN])
def test_transpose_with_even_data(data):
    n = len(data)
    # // = floor division
    permutation, perm_indices = PermutationHandler.transpose(data, 3)
    print(f"Permutation with even data: {permutation}")

@pytest.mark.parametrize("data", [TEST_DATA_LIST_ODD, TEST_DATA_NDARRAY_ODD])
def test_transpose_with_odd_data(data):
    n = len(data)
    # // = floor division
    permutation, p_indices = PermutationHandler.transpose(data, 2)
    print(f"Permutation with odd data: {permutation}")

@pytest.mark.parametrize("data", [TEST_DATA_LIST_EVEN, TEST_DATA_NDARRAY_EVEN])
def test_inverse_transpose_with_odd_data(data):
    n = len(data)
    # // = floor division
    permutation, perm_indices = PermutationHandler.transpose(data, 2)
    original_data = PermutationHandler.inverse_permute(permutation, perm_indices)

    assert np.array_equal(original_data, data)
    
@pytest.mark.parametrize("data", [TEST_DATA_LIST_EVEN, TEST_DATA_NDARRAY_EVEN])
def test_inverse_transpose_with_even_data(data):
    n = len(data)
    # // = floor division
    permutation, perm_indices = PermutationHandler.transpose(data, 3)
    original_data = PermutationHandler.inverse_permute(permutation, perm_indices)
    
    assert np.array_equal(original_data, data)



    



