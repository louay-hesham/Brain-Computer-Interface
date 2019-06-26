from emokit.headset import Headset
from prediction import extract_features
from headset_server.helper_functions import *
from django.http import HttpResponse
import json
import pickle
import xgboost as xgb
import numpy as np
import time
from CNN_classification import predict_cnn

model = pickle.load(open("local_model.dat", "rb"))
headset = Headset()

def is_ready(request):
    global headset
    if headset == None:
        headset = Headset()
        time.sleep(10)
    sample = headset.get_sample()
    if sample == None:
        headset = Headset()
        response = {
        "ready": "false",
        "sample": sample
        }
    else:
        response = {
        "ready": "true",
        "sample": sample
        }

    return HttpResponse(json.dumps(response))

def predict(request):
    global headset
    data = extract_data(request)
    samples_count = int(data["samples_count"])
    freq = int(data["freq"])
    delay = int(data["delay"])
    # print("Quering in", delay, "seconds")
    # time.sleep(delay)

    samples = headset.get_samples(samples_count, 1/freq, print_output=False)
    headset.stop()
    # samples_extra_features, label_testing = extract_features(samples, print_log=True)
    # samples_extra_features = np.array(samples_extra_features)
    # samples_extra_features = xgb.DMatrix(samples_extra_features)
    # predictions = model.predict(samples_extra_features)
    # print("Predictions array: ", end='')
    # print(predictions)
    # intent_labeling = np.array(['right_hand','left_hand', 'both_hands', 'both_feet', 'eye_closed'])
    # # intent_labeling = np.array(['','eye_closed', 'left_hand', 'right_hand', 'both_hands', 'both_feet'])
    #
    # labels_votes = {
    #     # '': 0,
    #     'right_hand': 0,
    #     'left_hand': 0,
    #     'both_hands': 0,
    #     'both_feet': 0,
    #     'eye_closed': 0
    # }
    # for pred in predictions:
    #     argmax = np.argmax(pred)
    #     labels_votes[intent_labeling[argmax]] += 1
    #
    # max_votes = 0
    # vote = 'right_hand'
    # for label, votes in labels_votes.items():
    #     if votes > max_votes:
    #         vote = label
    #         max_votes = votes
    #
    # print(labels_votes)
    # print(vote)

    predictions = predict_cnn(samples)
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

    response = {
    "prediction": vote
    }
    headset.start()
    return HttpResponse(json.dumps(response))
