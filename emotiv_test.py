import time
from emokit.headset import Headset

if __name__ == "__main__":
    with Headset() as headset:
        while True:
            samples = headset.get_samples_fft(10, 1/128, print_output=False)
            print("AF3\tF7\tF3\tFC5\tT7\tP7\tO1\tO2\tP8\tT8\tFC6\tF4\tF8\tAF4")
            for sample in samples:
                for channel in sample:
                    print(str(channel) + "\t", end='')
                print()
            print()
