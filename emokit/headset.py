import os

from .emotiv import Emotiv
from .sensors import sensors_mapping
from .util import system_platform

class Headset(Emotiv):

    def get_sensors_raw_data(self, print=False):
        packet = self.dequeue()
        if packet is None:
            return None
        else:
            self.print_raw_data(packet.sensors)
            return packet.sensors

    def print_raw_data_legacy(self):
        self.display_output = True

    def print_raw_data(self, sensor_data, clear=True):
        if clear:
            if system_platform == "Windows":
                os.system('cls')
            else:
                os.system('clear')
        output_template = """
        +=================================================================================================================================+
        |   AF3   |   F7   |   F3   |   FC5   |   T7   |   P7   |   O1   |   O2   |   P8   |   T8   |   FC6   |   F4   |   F8   |   AF4   |
        +---------+--------+--------+---------+--------+--------+--------+--------+--------+--------+---------+--------+--------+---------+
        |{AF3:^0.4f}|{F7:^0.3f}|{F3:^0.3f}|{FC5:^0.4f}|{T7:^0.3f}|{P7:^0.3f}|{O1:^0.3f}|{O2:^0.3f}|{P8:^0.3f}|{T8:^0.3f}|{FC6:^0.4f}|{F4:^0.3f}|{F8:^0.3f}|{AF4:^0.4f}|
        +=================================================================================================================================+
        """
        print(output_template.format(
            AF3 = sensor_data['AF3']['value'],
            F7 = sensor_data['F7']['value'],
            F3 = sensor_data['F3']['value'],
            FC5 = sensor_data['FC5']['value'],
            T7 = sensor_data['T7']['value'],
            P7 = sensor_data['P7']['value'],
            O1 = sensor_data['O1']['value'],
            O2 = sensor_data['O2']['value'],
            P8 = sensor_data['P8']['value'],
            T8 = sensor_data['T8']['value'],
            FC6 = sensor_data['FC6']['value'],
            F4 = sensor_data['F4']['value'],
            F8 = sensor_data['F8']['value'],
            AF4 = sensor_data['AF4']['value']
        ))
