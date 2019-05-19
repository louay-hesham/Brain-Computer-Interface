from emokit.headset import Headset
from prediction import extract_features
from headset_server.helper_functions import *
from django.http import HttpResponse
import json
import pickle
import xgboost as xgb
import numpy as np
import time

model = pickle.load(open("model.dat", "rb"))
headset = Headset()

def is_ready(request):
    global headset
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
    samples_count = data["samples_count"]
    freq = data["freq"]
    delay = data["delay"]
    print("Quering in", delay, "seconds")
    time.sleep(delay)

    samples = headset.get_samples_fft(samples_count, 1/freq, print_output=False)
    headset.stop()
    samples_extra_features, label_testing = extract_features(samples, print_log=True)
    samples_extra_features = np.array(samples_extra_features)
    samples_extra_features = xgb.DMatrix(samples_extra_features)
    predictions = model.predict(samples_extra_features)

    intent_labeling = np.array(['','eye_closed', 'left_hand', 'right_hand', 'both_hands', 'both_feet'])
    pred_argmax = np.argmax(predictions,1)
    # print(pred_argmax.shape)
    # print(pred_argmax)
    copy_pred = np.empty(predictions.shape, dtype=object)
    for i in range(predictions.shape[0]):
      copy_pred[i] = intent_labeling[pred_argmax[i]]
    print(copy_pred)

    response = {
    "prediction": copy_pred
    }
    headset.start()
    return HttpResponse(json.dumps(response))
