FT300_ID = '/dev/ttyUSB0'
FT300_BAUDRATE = 19200
FT300_FPS = 100
FT300_REGISTER_DICT = {
    "ProductionYear": 514,
    "SerialNumber0": 510,
    "SerialNumber1": 511,
    "SerialNumber2": 512,
    "SerialNumber3": 513,
    "Stream": 410,
    "F_x": 180,
    "F_y": 181,
    "F_z": 182,
    "M_x": 183,
    "M_y": 184,
    "M_z": 185,
    "acc_x": 190,
    "acc_y": 191,
    "acc_z": 192
}
FT300_FT_COEF = [100, 100, 100, 1000, 1000, 1000]
FT300_STREAM_FLAG = [0x200]
FT300_STREAM_START = bytes([0x20, 0x4e])
