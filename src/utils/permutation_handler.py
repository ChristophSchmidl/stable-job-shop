from typing import Union, Any, Optional, cast, List
import numpy as np
import torch


class PermutationHandler:
    '''
    This class is used to handle the permutations of different data types
    '''
    @staticmethod
    def permute(data: Union[List, np.ndarray], perm_indices: Optional[list[int]] = None) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
        '''
        Permutes data randomly or based on perm_indices (for repeating the same permutation)

        :param data: The data to permute
        :param perm_indices: The permutation indices. If None, the data will be permuted randomly
        :return: The permuted data, the permutation matrix (reverse permutation) and the permutation indices (for repeating the same permutation)
        :rtype: list[Union[List, np.ndarray]]
        '''
        data_len = len(data)
        permuted_data = None
        perm_matrix = None
        perm_indices = perm_indices

        if perm_indices is None:
            # Do random permutation
            perm_indices = torch.randperm(data_len).long()

        perm_matrix = torch.eye(data_len)[perm_indices].t()

        permuted_data = [ data[i] for i in perm_indices]
        return np.asarray(permuted_data), perm_matrix, np.asarray(perm_indices)

    def inverse_permute(data: list[Union[List, np.ndarray]], perm_matrix: Optional[list[int]] = None) -> np.ndarray:
        '''
        Reverts the already permuted data based on the permutation matrix

        :param data: The data to revert
        :param perm_matrix: The permutation matrix to revert the data
        :return: The reverted data
        :rtype: list[Union[List, np.ndarray]]
        '''
        data_len = len(data)
        reverted_data = None

        restore_indices = torch.Tensor(list(range(data_len))).view(data_len, 1)
        restore_indices = perm_matrix.mm(restore_indices).view(data_len).long()
        reverted_data = [ data[i] for i in restore_indices]

        return np.asarray(reverted_data)