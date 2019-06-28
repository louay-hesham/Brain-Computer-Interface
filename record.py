import time
import scipy.io as sio
import numpy as np
from emokit.headset import Headset
import os

if __name__ == "__main__":
    with Headset() as headset:
        for c in ['forward', 'backwards', 'up', 'down']:
            print("Class " + c)
            print("Recording in 10 seconds")
            time.sleep(7)
            print("Recording in 3 seconds")
            os.system("sudo env -i beep -f 400 -l 100")
            time.sleep(1)
            print("Recording in 2 seconds")
            os.system("sudo env -i beep -f 400 -l 100")
            time.sleep(1)
            print("Recording in 1 seconds")
            os.system("sudo env -i beep -f 400 -l 100")
            time.sleep(1)
            os.system("sudo env -i beep -f 550 -l 150")
            print("Recording")
            timeout = time.time() + 60
            total_data = []
            while time.time() < timeout:
                data = headset.get_sample(print_output=False)
                if data != None:
                    total_data.append(data)
                time.sleep(1/128)

            sio.savemat(c + ".mat", {"emotiv_7sub_5class": np.array(total_data)})
            os.system("sudo env -i beep -f 300 -l 100")
            print("10 seconds break")
            time.sleep(10)
