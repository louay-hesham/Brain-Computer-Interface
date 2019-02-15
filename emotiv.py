import time
from emokit.headset import Headset

if __name__ == "__main__":
    with Headset() as headset:
        while True:
            data = headset.get_sensors_raw_data(print=True)
            if data is not None:
                data
                # Do something with raw data here
            time.sleep(0.05)
