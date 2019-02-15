import os

from .emotiv import Emotiv
from .sensors import sensors_mapping
from .util import system_platform

class Headset(Emotiv):
    """
        Child class of emokit.emotiv.
        Adds extra functionality like extracting the actual sensor data and printing them.
        The old printing mechanism sometimes throws exceptions which will render the printing functionality useless.
    """

    def get_sensors_raw_data(self, print=False):
        packet = self.dequeue()
        if packet is None:
            return None
        else:
            if print:
                self.print_raw_data(packet.sensors)
            return packet.sensors

    def print_raw_data_legacy(self):
        self.display_output = True

    def limit_digits(self, num, length=8):
        return str(num)[0:length]

    def print_raw_data(self, sensor_data, clear=True):
        if clear:
            if system_platform == "Windows":
                os.system('cls')
            else:
                os.system('clear')

        output_template = """
        +=============================================================================================================================================================+
        |    AF3    |    F7    |    F3    |    FC5    |    T7    |    P7    |    O1    |    O2    |    P8    |    T8    |    FC6    |    F4    |    F8    |    AF4    |
        +-----------+----------+----------+-----------+----------+----------+----------+----------+----------+----------+-----------+----------+----------+-----------+
        | {AF3} | {F7} | {F3} | {FC5} | {T7} | {P7} | {O1} | {O2} | {P8} | {T8} | {FC6} | {F4} | {F8} | {AF4} |
        +=============================================================================================================================================================+
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
            AF4 = self.limit_digits(sensor_data['AF4']['value'], 9)
        ))
