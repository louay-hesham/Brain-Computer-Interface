import time
from emokit.headset import Headset

if __name__ == "__main__":
    total_data = []
    time.sleep(5)
    timeout = time.time() + 30
    with Headset(write=True) as headset:
        while time.time() < timeout:
            data = headset.get_sample(print_output=True)
            total_data.append(data)
            time.sleep(1/128)
        print(total_data)
