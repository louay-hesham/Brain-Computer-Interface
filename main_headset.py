import time
from emokit.headset import Headset
import pickle
import xgboost as xgb
import numpy as np
from prediction import extract_features
import scipy.io
import tensorflow as tf
from CNN_classification import predict_cnn


if __name__ == "__main__":
    # model = pickle.load(open("local_model.dat", "rb"))
    # with Headset() as headset:
    #     print("Quering in 10 seconds")
    #     time.sleep(10)
    #     samples = headset.get_samples(256, 1/128, print_output=False)
    # print("DATA")
    # print(data.shape)
    # samples = data.tolist()
    #samples_extra_features, label_testing = extract_features(samples, print_log=True)
    # print("before")
    # print(len(samples_extra_features))
    # print(len(samples_extra_features[0]))
    # samples_extra_features = np.array(samples_extra_features)
    # samples_extra_features = xgb.DMatrix(samples_extra_features)
    # predictions = model.predict(samples_extra_features)
    # print("samples_extra_features")
    # print(len(samples))
    # print(len(samples[0]))
    #
    # print(samples_extra_features)
    #
    # print ('predicting, classification error=%f' % ( (sum(int(predictions[i][j]) != label_testing[i][j]
    #                                                     for i in range((label_testing.shape[0]))
    #                                                     for j in range (label_testing.shape[1])) / float(len(label_testing)))))

    mat = scipy.io.loadmat("recordings/Louay2/right_hand.mat")
    print("I GOT THE MAT FILE")
    data = mat['emotiv_7sub_5class']
    data = data[:, 0:14]
    np.random.shuffle(data)
    predictions = predict_cnn(data)
    print("Predictions")
    print(predictions)
    print(len(predictions))

    intent_labeling = np.array(['right_hand','left_hand', 'both_hands', 'both_feet', 'eye_closed'])
    labels_votes = {
        'right_hand': 0,
        'left_hand': 0,
        'both_hands': 0,
        'both_feet': 0,
        'eye_closed': 0
    }

    for pred in predictions:
        labels_votes[intent_labeling[pred]] += 1

    max_votes = 0
    vote = 'right_hand'
    for label, votes in labels_votes.items():
        if votes > max_votes:
            vote = label
            max_votes = votes
    print(labels_votes)
    print(vote)
