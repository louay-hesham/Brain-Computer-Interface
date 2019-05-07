import os
import time

from scipy.fftpack import rfft
import numpy as np

from .emotiv import Emotiv
from .sensors import sensors_mapping
from .util import system_platform

class Headset(Emotiv):
    """
        Child class of emokit.emotiv.
        Adds extra functionality like extracting the actual sensor data and printing them.
        The old printing mechanism sometimes throws exceptions which will render the printing functionality useless.
    """

    channels_order = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    def get_samples_fft(self, n_samples, period, print_output=False):
        samples = []
        sample = None
        while sample == None:
            sample = self.get_sample(print_output)

        for i in range(0,n_samples + 1):
            sample = self.get_sample(print_output)
            if not sample == None:
                samples.append(sample)
                time.sleep(period)
        transposed = [list(i) for i in zip(*samples)] ## Transpose so each sub-array is the data of a channel in time domain
        fft_transposed = [rfft(np.array(i)) for i in transposed] ## Compute FFT (real values)
        fft_samples = [list(np.round(i, 0)) for i in zip(*fft_transposed)] ## Transpose again so each channel is in a column instead of a row, rounds to the nearest unit while transposing
        return fft_samples[1:]

    def get_sample(self, print_output=False):
        data = self.get_sensors_raw_data(print_output)
        if data is None:
            return None
        return [data[channel]['value'] for channel in Headset.channels_order]

    def get_sensors_raw_data(self, print_output=False):
        packet = self.dequeue()
        if packet is None:
            return None
        else:
            if print_output:
                self.print_raw_data(packet.sensors)
            return packet.sensors

    def print_raw_data_legacy(self):
        self.display_output = True

    def limit_digits(self, num, length=8):
        num_s = str(num)[0:length]
        if len(num_s) < length:
            num_s += '0' * (length - len(num_s))
        return num_s

    def print_raw_data(self, sensor_data, clear=True):
        if clear:
            if system_platform == "Windows":
                os.system('cls')
            else:
                os.system('clear')

        output_template = """
        +=========================================================================================================================================================================+
        | Property  |    AF3    |    F7    |    F3    |    FC5    |    T7    |    P7    |    O1    |    O2    |    P8    |    T8    |    FC6    |    F4    |    F8    |    AF4    |
        +-----------+----------+----------+-----------+----------+----------+----------+----------+----------+----------+-----------+----------+----------+-----------+-----------+
        |   Value   | {AF3} | {F7} | {F3} | {FC5} | {T7} | {P7} | {O1} | {O2} | {P8} | {T8} | {FC6} | {F4} | {F8} | {AF4} |
        |  Quality  | {AF3_q} | {F7_q} | {F3_q} | {FC5_q} | {T7_q} | {P7_q} | {O1_q} | {O2_q} | {P8_q} | {T8_q} | {FC6_q} | {F4_q} | {F8_q} | {AF4_q} |
        +=========================================================================================================================================================================+
        """
        print(output_template.format(
            AF3 = self.limit_digits(sensor_data['AF3']['value'], 9),
            F7 = self.limit_digits(self.limit_digits(sensor_data['F7']['value'])),
            F3 = self.limit_digits(sensor_data['F3']['value']),
            FC5 = self.limit_digits(sensor_data['FC5']['value'], 9),
            T7 = self.limit_digits(sensor_data['T7']['value']),
            P7 = self.limit_digits(sensor_data['P7']['value']),
            O1 = self.limit_digits(sensor_data['O1']['value']),
            O2 = self.limit_digits(sensor_data['O2']['value']),
            P8 = self.limit_digits(sensor_data['P8']['value']),
            T8 = self.limit_digits(sensor_data['T8']['value']),
            FC6 = self.limit_digits(sensor_data['FC6']['value'], 9),
            F4 = self.limit_digits(sensor_data['F4']['value']),
            F8 = self.limit_digits(sensor_data['F8']['value']),
            AF4 = self.limit_digits(sensor_data['AF4']['value'], 9),

            AF3_q =  self.limit_digits(sensor_data['AF3']['quality'], 9),
            F7_q =  self.limit_digits(self.limit_digits(sensor_data['F7']['quality'])),
            F3_q =  self.limit_digits(sensor_data['F3']['quality']),
            FC5_q =  self.limit_digits(sensor_data['FC5']['quality'], 9),
            T7_q =  self.limit_digits(sensor_data['T7']['quality']),
            P7_q =  self.limit_digits(sensor_data['P7']['quality']),
            O1_q =  self.limit_digits(sensor_data['O1']['quality']),
            O2_q =  self.limit_digits(sensor_data['O2']['quality']),
            P8_q =  self.limit_digits(sensor_data['P8']['quality']),
            T8_q =  self.limit_digits(sensor_data['T8']['quality']),
            FC6_q =  self.limit_digits(sensor_data['FC6']['quality'], 9),
            F4_q =  self.limit_digits(sensor_data['F4']['quality']),
            F8_q =  self.limit_digits(sensor_data['F8']['quality']),
            AF4_q =  self.limit_digits(sensor_data['AF4']['quality'], 9)
        ))
