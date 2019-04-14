import time
from emokit.headset import Headset

if __name__ == "__main__":
    with Headset() as headset:
        while True:
            sample = headset.get_sample(print=True)
            if sample is not None:
                # print(data)
                sample
                # Do something with raw data here
            time.sleep(0.05)
