import time
from emokit.headset import Headset
import pickle
import xgboost as xgb
import numpy as np
from prediction import extract_features

if __name__ == "__main__":
    model = pickle.load(open("model.dat", "rb"))
    with Headset() as headset:
        samples = headset.get_samples_fft(1, 1/128, print_output=False)
    samples_extra_features = extract_features(samples, print_log=True)
    samples_extra_features = np.array(samples_extra_features)
    samples_extra_features = xgb.DMatrix(samples_extra_features)
    predictions = model.predict(samples_extra_features)
    print(predictions)
