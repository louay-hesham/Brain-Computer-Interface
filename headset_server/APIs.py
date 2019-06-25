from emokit.headset import Headset
from prediction import extract_features
from headset_server.helper_functions import *
from django.http import HttpResponse
import json
import pickle
import xgboost as xgb
import numpy as np
import time

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
    print("Quering in", delay, "seconds")
    time.sleep(delay)

    samples = headset.get_samples(samples_count, 1/freq, print_output=False)
    headset.stop()
    samples_extra_features, label_testing = extract_features(samples, print_log=True)
    samples_extra_features = np.array(samples_extra_features)
    samples_extra_features = xgb.DMatrix(samples_extra_features)
    predictions = model.predict(samples_extra_features)
    print("Predictions array: ", end='')
    print(predictions)
    intent_labeling = np.array(['right_hand','left_hand', 'both_hands', 'both_feet', 'eye_closed'])
    pred_argmax = np.argmax(predictions,1)
    copy_pred = np.empty(predictions.shape, dtype=object)
    for i in range(predictions.shape[0]):
      copy_pred[i] = intent_labeling[pred_argmax[i]]
    unique,pos = np.unique(copy_pred,return_inverse=True)
    counts = np.bincount(pos)
    maxpos = counts.argmax()

    response = {
    "prediction": unique[maxpos]
    }
    headset.start()
    return HttpResponse(json.dumps(response))
