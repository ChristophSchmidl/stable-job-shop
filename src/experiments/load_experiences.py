import numpy as np

'''
Evaluate the integrity of the collected dataset:

1. Before any action has been taken: The first position of every sub-array should be 1, indicating that the job is available
    a) Check that the action_mask is True for every job but False for the no-op operation (last index)
2. If an action has been taken, check the index of that action in the next step of the array
    a) Check that the action_mask is False for that specific position
    b) Check that there is a differce between the value of the previous step array and the current one
'''

# Check the first three steps through the environment, 
# if the changed values in the array match the action taken
# during the prior step
STEPS = 3
DATA_DIR = "data/experiences/30mins_tuned_policy"


def exactly_one_non_zero(lst):
    count = 0
    for item in lst:
        if item != 0:
            count += 1
            if count > 1:
                return False
    return count == 1

def check_every_first_element_is_one(arr):
    for row in arr:
        if row[0] != 1:
            return False
    return True

def check_all_true_except_last(arr):
    return all(arr[:-1]) and not arr[-1]

def check_if_idx_is_true(arr, idx):
    return arr[idx] == True

def check_if_idx_is_false(arr, idx):
    print(arr[idx])
    return arr[idx] == False

def print_info(data, idx):
    print(f"State: {data['states'][idx]}") # wrong permutation?
    print(data["actions"][idx])
    print(data["action_masks"][idx]) # wrong permutation?
    print(data["dones"][idx])
    print(data["episodes"][idx])

    print(f"Shape of states: {data['states'].shape}")
    print(f"Shape of actions: {data['actions'].shape}")
    print(f"Shape of action_masks: {data['action_masks'].shape}")
    print(f"Shape of dones: {data['dones'].shape}")

def evaluate_data(data_path):
    data = np.load(data_path)

    prev_action = None
    for idx in range(STEPS):
        #print_info(data, idx)
        if idx == 0:
            prev_action = data["actions"][idx].item()
            assert check_every_first_element_is_one(data['states'][idx]), "First element is not 1.0"
            assert check_all_true_except_last(data["action_masks"][idx]), "Last element is not false"
        if idx != 0:
            if prev_action is not None:
                #print(data["action_masks"][idx])
                assert check_if_idx_is_false(data["action_masks"][idx], prev_action), "Index is not false"
                assert exactly_one_non_zero(data['states'][idx][prev_action]), "Exactly one non zero failed"

            prev_action = data["actions"][idx].item()

    print(f"Evaluation successful for data {data_path}")

#data = np.load(f"{DATA_DIR}/experiences_random_100-episodes.npz")
#data = np.load(f"{DATA_DIR}/experiences_transpose-8_1000-episodes.npz")
#data = np.load(f"{DATA_DIR}/experiences_no-permutation_1000-episodes.npz")

data_paths = []
data_paths.append("./data/experiences/30mins_tuned_policy/experiences_no-permutation_1000-episodes.npz")
data_paths.append("./data/experiences/30mins_tuned_policy/experiences_random_1000-episodes.npz")
transpose_paths = [f"./data/experiences/30mins_tuned_policy/experiences_transpose-{index}_1000-episodes.npz" for index in range(1,16)]
data_paths.extend(transpose_paths)

for data_path in data_paths:
    evaluate_data(data_path)








