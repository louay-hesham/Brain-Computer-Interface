import time
from emokit.headset import Headset

if __name__ == "__main__":
    with Headset(write=True) as headset:
        while True:
            data = headset.get_sample(print_output=True)
            time.sleep(1/128)
