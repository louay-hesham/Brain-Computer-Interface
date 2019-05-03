import time
from emokit.headset import Headset
import pickle
import xgboost as xgb
import numpy as np

if __name__ == "__main__":
    with Headset() as headset:
        model = pickle.load(open("model.dat", "rb"))
        while True:
            sample = headset.get_sample(print=True)
            if sample is not None:
                sample = np.array([sample])
                sample = xgb.DMatrix(sample)
                # prediction = model.predict(sample)
                print()
            time.sleep(0.05)
