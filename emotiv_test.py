import time
from emokit.headset import Headset

if __name__ == "__main__":
    with Headset(write=True, force_old_crypto=True) as headset:
        while True:
            data = headset.get_sensors_raw_data(print_output=True)
            if data is not None:
                data
                # Do something with raw data here
            time.sleep(0.05)
