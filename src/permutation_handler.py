from typing import Union, Any, Optional, cast, List
import random
import numpy as np
import torch


class PermutationHandler:
    '''
    This class is used to handle the permutations of different data types
    '''
    @staticmethod
    def permute(data: Union[List, np.ndarray], perm_indices: Optional[list[int]] = None) -> tuple[np.ndarray, np.ndarray]:
        '''
        Permutes data randomly or based on perm_indices (for repeating the same permutation)

        :param data: The data to permute
        :param perm_indices: The permutation indices. If None, the data will be permuted randomly
        :return: The permuted data, the permutation matrix (reverse permutation) and the permutation indices (for repeating the same permutation)
        :rtype: list[Union[torch.Tensor, np.ndarray]]
        '''
        data = data.copy()
        data_len = len(data)
        permuted_data = None
        perm_matrix = None
        perm_indices = perm_indices

        if perm_indices is None:
            # Do random permutation
            perm_indices = torch.randperm(data_len).long()

        #perm_matrix = torch.eye(data_len)[perm_indices].t()

        permuted_data = [ data[i] for i in perm_indices]
        #return np.asarray(permuted_data), perm_matrix, np.asarray(perm_indices)
        return np.asarray(permuted_data), np.asarray(perm_indices)

    @staticmethod
    def inverse_permute(data: list[Union[List, np.ndarray]], perm_indices: Optional[list[int]] = None) -> np.ndarray:
        '''
        Reverts the already permuted data based on the permutation matrix

        :param data: The data to revert
        :param perm_indices: The permutation indices to revert the data with np.argsort
        :return: The reverted data
        :rtype: np.ndarray
        '''
        data = data.copy()
        reverted_data = None
        inverse_indices = np.argsort(perm_indices)
        reverted_data = [ data[i] for i in inverse_indices]

        return np.asarray(reverted_data)

    @staticmethod
    def old_inverse_permute(data: list[Union[List, np.ndarray]], perm_matrix: Optional[list[int]] = None) -> np.ndarray:
        '''
        Reverts the already permuted data based on the permutation matrix

        :param data: The data to revert
        :param perm_matrix: The permutation matrix to revert the data
        :return: The reverted data
        :rtype: np.ndarray
        '''
        data = data.copy()
        data_len = len(data)
        reverted_data = None

        restore_indices = torch.Tensor(list(range(data_len))).view(data_len, 1)
        restore_indices = perm_matrix.mm(restore_indices).view(data_len).long()
        reverted_data = [ data[i] for i in restore_indices]

        return np.asarray(reverted_data)

    @staticmethod
    def transpose(data: list[Union[List, np.ndarray]], n_swaps: int) -> tuple[np.ndarray, np.ndarray]:
        '''
        A transposition is a type of permutation where the elements are swapped in pairs. 
        In this implementation, once a pair is swapped, it cannot be swapped again.

        https://martin-thoma.com/permutationen-und-transpositionen/

        :param data: The data to transpose
        :param n_swaps: How many swaps should be performed
        :return: The transposed data and the transposition/permutation indices
        :rtype: Union[List, np.ndarray]
        '''
        if n_swaps > len(data)//2:
            raise ValueError("Distance cannot be greater than half of the length of the list")

        data = data.copy()
        permutation = None

        # Transform to ndarray if list
        if type(data) is np.ndarray:
            permutation = data.tolist()
        else:
            # if already a list, just copy
            permutation = data.copy()

        orig_indices = list(range(len(data)))
        # [0,1,2,3,4,5]
        permuted_indices = list(range(len(data)))

        for i in range(n_swaps):
            # Pick a random index of the original indices
            index_1 = random.choice(orig_indices)
            # Remove the index from the original indices
            orig_indices.remove(index_1)

            # Pick a second random index of the original indices
            index_2 = random.choice(orig_indices)
            # Remove the index from the original indices
            orig_indices.remove(index_2)

            permuted_indices[index_1], permuted_indices[index_2] = permuted_indices[index_2], permuted_indices[index_1]
            permutation[index_1], permutation[index_2] = permutation[index_2], permutation[index_1]

        return np.asarray(permutation), np.asarray(permuted_indices)

    @staticmethod
    def get_permuted_action_index(action: int, perm_indices, include_no_op=True):
        '''
        Usage: When the policy predicts a certain action, we need to get the index of that action in the permuted action space
               and collect it together with the permuted observation     
        Note: the length of perm_indices already tells us how many jobs there are.
              However, we need to increase it by one to account for the no-op operation
        Returns the permuted action index based on the permutation indices
        '''
        assert isinstance(action, (int, integer)), "Action must be an integer"

        if len(perm_indices) == 1:
            perm_indices = perm_indices[0]

        action_space = len(perm_indices)
        

        # If the action index is the last action in the array
        # then it's no-op action and we return the no-op action immediately
        if include_no_op and action == action_space:
            return action

        return np.where(perm_indices == action)[0][0]

    @staticmethod
    def get_inverse_permuted_action_index(action: int, perm_indices, include_no_op=True):
        '''
        Usage: When the policy predicts a certain action, we need to get the index of that action in the permuted action space
               and collect it together with the permuted observation     
        Note: the length of perm_indices already tells us how many jobs there are.
              However, we need to increase it by one to account for the no-op operation
        Returns the permuted action index based on the permutation indices
        '''
        action = action[0]
        assert np.issubdtype(action, integer), "Action must be an integer"

        action_space = len(perm_indices)

        # If the action index is the last action in the array
        # then it's no-op action and we return the no-op action immediately
        if include_no_op and action == action_space:
            return action
        else:
            inverse_indices = np.argsort(perm_indices)
            return np.where(inverse_indices == action)[0][0]

    @staticmethod
    def inverse_action_mask(action_mask, perm_indices):
        action_mask = np.copy(action_mask)

        # original_action_mask[0] = the raw action mask
        # original_action_mask[0][:-1] = the raw action mask without the no_op
                
        inverse_action_mask = PermutationHandler.inverse_permute(action_mask[:-1], perm_indices)
        inverse_action_mask = np.append(inverse_action_mask,action_mask[-1]).astype(action_mask.dtype) # Add the no-op

        return inverse_action_mask

    @staticmethod
    def permute_action_mask(action_mask, perm_indices):
        original_action_mask = np.copy(action_mask)
                
        permuted_action_mask, _ = PermutationHandler.permute(original_action_mask[:-1], perm_indices)
        permuted_action_mask = np.append(permuted_action_mask,original_action_mask[-1]).astype(action_mask.dtype) # Add the no-op

        return permuted_action_mask

    