import time
from emokit.headset import Headset

if __name__ == "__main__":
    print("Starting in 10 seconds")
    time.sleep(7)
    print("Starting in 3 seconds")
    time.sleep(1)
    print("Starting in 2 seconds")
    time.sleep(1)
    print("Starting in 1 seconds")
    time.sleep(1)
    with Headset(write=True) as headset:
        while True:
            data = headset.get_sample(print_output=True)
            time.sleep(1/128)
