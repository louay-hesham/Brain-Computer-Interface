import time
import scipy.io as sio
import numpy as np
from emokit.headset import Headset

if __name__ == "__main__":
    with Headset() as headset:
        for c in ['left_hand', 'right_hand', 'both_hands', 'both_feet', 'eye_closed']:
            print("Class " + c)
            print("Recording in 10 seconds")
            time.sleep(7)
            print("Recording in 3 seconds")
            time.sleep(1)
            print("Recording in 2 seconds")
            time.sleep(1)
            print("Recording in 1 seconds")
            time.sleep(1)
            print("Recording")
            timeout = time.time() + 60
            total_data = []
            while time.time() < timeout:
                data = headset.get_sample(print_output=False)
                if data != None:
                    total_data.append(data)
                time.sleep(1/128)

            sio.savemat(c + ".mat", {"emotiv_7sub_5class": np.array(total_data)})
            print("10 seconds break")
            time.sleep(10)
