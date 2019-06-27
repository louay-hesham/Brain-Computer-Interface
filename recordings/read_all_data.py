import scipy.io
import numpy as np

# right_hand_mat = scipy.io.loadmat("recordings/Louay2/right_hand.mat")
# right_hand_data = right_hand_mat['emotiv_7sub_5class']

# left_hand_mat = scipy.io.loadmat("recordings/Louay2/left_hand.mat")
# left_hand_data = left_hand_mat['emotiv_7sub_5class']

both_hands_mat = scipy.io.loadmat("recordings/Louay2/both_hands.mat")
both_hands_data = both_hands_mat['emotiv_7sub_5class']

# both_feet_mat = scipy.io.loadmat("recordings/Louay2/both_feet.mat")
# both_feet_data = both_feet_mat['emotiv_7sub_5class']

eye_closed_mat = scipy.io.loadmat("recordings/Louay2/eye_closed.mat")
eye_closed_data = eye_closed_mat['emotiv_7sub_5class']


# right_hand_labels = np.full((right_hand_data.shape[0], 1), 0)
# right_hand_data = np.append(right_hand_data, right_hand_labels, axis=1)

# left_hand_labels = np.full((left_hand_data.shape[0], 1), 1)
# left_hand_data = np.append(left_hand_data, left_hand_labels, axis=1)

both_hands_labels = np.full((both_hands_data.shape[0], 1), 0)
both_hands_data = np.append(both_hands_data, both_hands_labels, axis=1)

# both_feet_labels = np.full((both_feet_data.shape[0], 1), 3)
# both_feet_data = np.append(both_feet_data, both_feet_labels, axis=1)

eye_closed_labels = np.full((eye_closed_data.shape[0], 1), 2)
eye_closed_data = np.append(eye_closed_data, eye_closed_labels, axis=1)

# right_hand_data = np.append(right_hand_data, left_hand_data, axis=0)
# right_hand_data = np.append(right_hand_data, both_hands_data, axis=0)
# right_hand_data = np.append(right_hand_data, both_feet_data, axis=0)
both_hands_data = np.append(both_hands_data, eye_closed_data, axis=0)
data = both_hands_data

print(data.shape)
scipy.io.savemat("louay_eyes_hands.mat", {'Eddeny3a2lk':data})
