# -*- coding: utf-8 -*-
# This is an example of popping a packet from the Emotiv class's packet queue


import time

from emokit.emotiv import Emotiv
from emokit.sensors import sensors_mapping

if __name__ == "__main__":
    with Emotiv() as headset:
        while True:
            packet = headset.dequeue()
            if packet is not None:
                print(packet.raw_data)
                # Do something with the raw data here
                pass
            time.sleep(0.001)
