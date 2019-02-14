# -*- coding: utf-8 -*-

sensors_14_bits = {
    'F3': [10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7],  # <- 8
    'FC5': [28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9],  # <- 2
    'AF3': [46, 47, 32, 33, 34, 35, 36, 37, 38, 39, 24, 25, 26, 27],  # <- 4
    'F7': [48, 49, 50, 51, 52, 53, 54, 55, 40, 41, 42, 43, 44, 45],  # <-6
    'T7': [66, 67, 68, 69, 70, 71, 56, 57, 58, 59, 60, 61, 62, 63],  # <- 8
    'P7': [84, 85, 86, 87, 72, 73, 74, 75, 76, 77, 78, 79, 64, 65],  # <- 10
    'O1': [102, 103, 88, 89, 90, 91, 92, 93, 94, 95, 80, 81, 82, 83],  # <- 12
    'O2': [140, 141, 142, 143, 128, 129, 130, 131, 132, 133, 134, 135, 120, 121],  # <- 2
    'P8': [158, 159, 144, 145, 146, 147, 148, 149, 150, 151, 136, 137, 138, 139],  # <- 4
    'T8': [160, 161, 162, 163, 164, 165, 166, 167, 152, 153, 154, 155, 156, 157],  # <- 6
    'F8': [178, 179, 180, 181, 182, 183, 168, 169, 170, 171, 172, 173, 174, 175],  # <- 8
    'AF4': [196, 197, 198, 199, 184, 185, 186, 187, 188, 189, 190, 191, 176, 177],  # <- 10
    'FC6': [214, 215, 200, 201, 202, 203, 204, 205, 206, 207, 192, 193, 194, 195],  # <- 12
    'F4': [216, 217, 218, 219, 220, 221, 222, 223, 208, 209, 210, 211, 212, 213],  # <- 6
    'GYRO_X': [224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239],  # 232, 233, 234, 235],
    'GYRO_Y': [248, 249, 250, 251, 252, 253, 254, 255, 240, 241, 242, 243, 244, 245, 246, 247],
    # , 235, 234, 233, 232],  # 232, 233, 234, 235, 236, 237, 238, 239]
    # 'GYRO_X_2': [236, 237, 238, 239, 244, 245, 246, 247],
    # 'GYRO_X_3': [244, 245, 246, 247, 236, 237, 238, 239],
    # 'GYRO_X_4': [240, 241, 242, 243, 236, 237, 238, 239],
    # 'GYRO_X_5': [240, 241, 242, 243, 232, 233, 234, 235],
    # 'GYRO_X_6': [236, 237, 238, 239, 232, 233, 234, 235],#244, 245, 246, 247],
    # 'GYRO_Y_1': [240, 241, 242, 243, 244, 245, 246, 247],#244, 245, 246, 247],
    # 'GYRO_Y': [244, 245, 246, 247, 240, 241, 242, 243], #, 235, 234, 233, 232],  # 232, 233, 234, 235, 236, 237, 238, 239]
    # 'GYRO_Y_3': [236, 237, 238, 239, 240, 241, 242, 243],
    # 'GYRO_Y_4': [232, 233, 234, 235, 240, 241, 242, 243],#244, 245, 246, 247],
    # 'GYRO_Y_5': [240, 241, 242, 243, 244, 245, 246, 247],
    #'GYRO_Y': [240, 241, 242, 243, 236, 237, 238, 239],  # 232, 233, 234, 235, 236, 237, 238, 239]
    # 'GYRO_Y': [236, 237, 238, 239, 232, 233, 234, 235],
    # 'GYRO_X': [244, 245, 246, 247, 240, 241, 242, 243]
}

# THESE ARE NOT CORRECT, MOST LIKELY
# TODO: FIX THIS
sensors_16_bits = {
    'F3': [9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    'FC5': [27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10],
    'AF3': [45, 46, 47, 32, 33, 34, 35, 36, 37, 38, 39, 24, 25, 26, 27, 28],
    'F7': [47, 48, 49, 50, 51, 52, 53, 54, 55, 40, 41, 42, 43, 44, 45, 46],
    'T7': [65, 66, 67, 68, 69, 70, 71, 56, 57, 58, 59, 60, 61, 62, 63, 64],
    'P7': [83, 84, 85, 86, 87, 72, 73, 74, 75, 76, 77, 78, 79, 64, 65, 66],
    'O1': [101, 102, 103, 88, 89, 90, 91, 92, 93, 94, 95, 80, 81, 82, 83, 84],
    'O2': [139, 140, 141, 142, 143, 128, 129, 130, 131, 132, 133, 134, 135, 120, 121, 122],
    'P8': [157, 158, 159, 144, 145, 146, 147, 148, 149, 150, 151, 136, 137, 138, 139, 140],
    'T8': [159, 160, 161, 162, 163, 164, 165, 166, 167, 152, 153, 154, 155, 156, 157, 158],
    'F8': [177, 178, 179, 180, 181, 182, 183, 168, 169, 170, 171, 172, 173, 174, 175, 176],
    'AF4': [195, 196, 197, 198, 199, 184, 185, 186, 187, 188, 189, 190, 191, 176, 177, 178],
    'FC6': [213, 214, 215, 200, 201, 202, 203, 204, 205, 206, 207, 192, 193, 194, 195, 194],
    'F4': [215, 216, 217, 218, 219, 220, 221, 222, 223, 208, 209, 210, 211, 212, 213, 214]
}

sensors_16_bytes = {
    'F3': [2, 3],
    'FC5': [4, 5],
    'AF3': [6, 7],
    'F7': [8, 9],
    'T7': [10, 11],
    'P7': [12, 13],
    'O1': [14, 15],
    'O2': [18, 19],
    'P8': [20, 21],
    'T8': [22, 23],
    'F8': [24, 25],
    'AF4': [26, 27],
    'FC6': [28, 29],
    'F4': [30, 31],
    # 'QUALITY': [16, 17]
}

quality_bits = [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]

sensor_quality_bit = {
    0: "F3",
    64: 'F3',
    1: 'FC5',
    65: 'FC5',
    2: 'AF3',
    66: 'AF3',
    3: 'F7',
    67: 'F7',
    4: 'T7',
    68: 'T7',
    5: 'P7',
    69: 'P7',
    6: 'O1',
    70: 'O1',
    7: 'O2',
    71: 'O2',
    8: 'P8',
    72: 'P8',
    9: 'T8',
    73: 'T8',
    10: 'F8',
    74: 'F8',
    11: 'AF4',
    75: 'AF4',
    12: 'FC6',
    76: 'FC6',
    80: 'FC6',
    13: 'F4',
    77: 'F4',
    14: 'F8',
    78: 'F8',
    15: 'AF4',
    79: 'AF4'
}

sensors_mapping = {
    'F3': {'value': 0, 'quality': 0},
    'FC6': {'value': 0, 'quality': 0},
    'P7': {'value': 0, 'quality': 0},
    'T8': {'value': 0, 'quality': 0},
    'F7': {'value': 0, 'quality': 0},
    'F8': {'value': 0, 'quality': 0},
    'T7': {'value': 0, 'quality': 0},
    'P8': {'value': 0, 'quality': 0},
    'AF4': {'value': 0, 'quality': 0},
    'F4': {'value': 0, 'quality': 0},
    'AF3': {'value': 0, 'quality': 0},
    'O2': {'value': 0, 'quality': 0},
    'O1': {'value': 0, 'quality': 0},
    'FC5': {'value': 0, 'quality': 0},
    'X': {'value': 0, 'quality': 0},
    'Y': {'value': 0, 'quality': 0},
    'Z': {'value': '?', 'quality': 0},
    'Unknown': {'value': 0, 'quality': 0}
}

# this is useful for further reverse engineering for EmotivPacket
byte_names = {
    "saltie-sdk": [  # also clamshell-v1.3-sydney
        "INTERPOLATED",
        "COUNTER",
        "BATTERY",
        "FC6",
        "F8",
        "T8",
        "PO4",
        "F4",
        "AF4",
        "FP2",
        "OZ",
        "P8",
        "FP1",
        "AF3",
        "F3",
        "P7",
        "T7",
        "F7",
        "FC5",
        "GYRO_X",
        "GYRO_Y",
        "RESERVED",
        "ETE1",
        "ETE2",
        "ETE3",
    ],
    "clamshell-v1.3-san-francisco": [  # amadi ?
        "INTERPOLATED",
        "COUNTER",
        "BATTERY",
        "F8",
        "UNUSED",
        "AF4",
        "T8",
        "UNUSED",
        "T7",
        "F7",
        "F3",
        "F4",
        "P8",
        "PO4",
        "FC6",
        "P7",
        "AF3",
        "FC5",
        "OZ",
        "GYRO_X",
        "GYRO_Y",
        "RESERVED",
        "ETE1",
        "ETE2",
        "ETE3",
    ],
    "clamshell-v1.5": [
        "INTERPOLATED",
        "COUNTER",
        "BATTERY",
        "F3",
        "FC5",
        "AF3",
        "F7",
        "T7",
        "P7",
        "O1",
        "SQ_WAVE",
        "UNUSED",
        "O2",
        "P8",
        "T8",
        "F8",
        "AF4",
        "FC6",
        "F4",
        "GYRO_X",
        "GYRO_Y",
        "RESERVED",
        "ETE1",
        "ETE2",
        "ETE3",
    ],
    "clamshell-v3.0": [
        "INTERPOLATED",
        "COUNTER",
        "BATTERY",
        "F3",
        "FC5",
        "AF3",
        "F7",
        "T7",
        "P7",
        "O1",
        "SQ_WAVE",
        "UNUSED",
        "O2",
        "P8",
        "T8",
        "F8",
        "AF4",
        "FC6",
        "F4",
        "GYRO_X",
        "GYRO_Y",
        "RESERVED",
        "ETE1",
        "ETE2",
        "ETE3",
    ],
}
