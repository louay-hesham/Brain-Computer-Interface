import time
from emokit.headset import Headset
import pickle
import xgboost as xgb
import numpy as np
from prediction import extract_features

if __name__ == "__main__":
    model = pickle.load(open("pima.pickle.dat", "rb"))
    with Headset() as headset:
        print("Quering in 10 seconds")
        time.sleep(10)
        samples = headset.get_samples_fft(256, 1/128, print_output=False)
    samples_extra_features, label_testing = extract_features(samples, print_log=True)
    samples_extra_features = np.array(samples_extra_features)
    samples_extra_features = xgb.DMatrix(samples_extra_features)
    predictions = model.predict(samples_extra_features)

    print ('predicting, classification error=%f' % ( (sum(int(predictions[i][j]) != label_testing[i][j]
                                                        for i in range((label_testing.shape[0]))
                                                        for j in range (label_testing.shape[1])) / float(len(label_testing)))))

    intent_labeling = np.array(['', 'up', 'down', 'left', 'right', 'middle', 'eyes_closed'])
    pred_argmax = np.argmax(predictions,1)
    print(pred_argmax.shape)
    print(pred_argmax)
    copy_pred = np.empty(predictions.shape, dtype=object)
    for i in range(predictions.shape[0]):
      copy_pred[i] = intent_labeling[pred_argmax[i]]
    print(copy_pred)
