import scipy.io
import numpy as np

right_hand_mat = scipy.io.loadmat("recordings/Louay/right_hand.mat")
right_hand_data = right_hand_mat['emotiv_7sub_5class']
right_hand_data = right_hand_data[0:499, :]

left_hand_mat = scipy.io.loadmat("recordings/Louay/left_hand.mat")
left_hand_data = left_hand_mat['emotiv_7sub_5class']
left_hand_data = left_hand_data[0:499, :]

both_hands_mat = scipy.io.loadmat("recordings/Louay/both_hands.mat")
both_hands_data = both_hands_mat['emotiv_7sub_5class']
both_hands_data = both_hands_data[0:499, :]

both_feet_mat = scipy.io.loadmat("recordings/Louay/both_feet.mat")
both_feet_data = both_feet_mat['emotiv_7sub_5class']
both_feet_data = both_feet_data[0:499, :]

eye_closed_mat = scipy.io.loadmat("recordings/Louay/eye_closed.mat")
eye_closed_data = eye_closed_mat['emotiv_7sub_5class']
eye_closed_data = eye_closed_data[0:499, :]


right_hand_labels = np.full((right_hand_data.shape[0], 1), 0)
right_hand_data = np.append(right_hand_data, right_hand_labels, axis=1)

left_hand_labels = np.full((left_hand_data.shape[0], 1), 1)
left_hand_data = np.append(left_hand_data, left_hand_labels, axis=1)

both_hands_labels = np.full((both_hands_data.shape[0], 1), 2)
both_hands_data = np.append(both_hands_data, both_hands_labels, axis=1)

both_feet_labels = np.full((both_feet_data.shape[0], 1), 3)
both_feet_data = np.append(both_feet_data, both_feet_labels, axis=1)

eye_closed_labels = np.full((eye_closed_data.shape[0], 1), 4)
eye_closed_data = np.append(eye_closed_data, eye_closed_labels, axis=1)
print(right_hand_data.shape)
print(left_hand_data.shape)

right_hand_data = np.append(right_hand_data, left_hand_data, axis=0)
right_hand_data = np.append(right_hand_data, both_hands_data, axis=0)
right_hand_data = np.append(right_hand_data, both_feet_data, axis=0)
right_hand_data = np.append(right_hand_data, eye_closed_data, axis=0)
data = right_hand_data
scipy.io.savemat("data_500_samples.mat", {'Eddeny3a2lk':data})


print(data.shape)

# full_data_mat = scipy.io.loadmat("localdata.mat")
# full_data = full_data_mat['Eddeny3a2lk']
# full_data = np.append(full_data, data, axis=0)
# print(full_data)
# scipy.io.savemat("full_local_data", {'Eddeny3a2lk':full_data})
# print("fasl")
# temp = scipy.io.loadmat("full_local_data.mat")
# temp_data = temp['Eddeny3a2lk']
# print(temp_data.shape)
