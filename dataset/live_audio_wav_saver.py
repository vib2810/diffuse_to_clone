import os
import numpy as np
from scipy.io.wavfile import write
count = 0

def convert_uint8_to_int16(data):
    # Center uint8 data around zero (128 becomes 0, 0 becomes -128, 255 becomes 127)
    data = data.astype(np.float32) - 128
    # Scale the centered data to the int16 range
    data = data / 127 * 32767
    return data.astype(np.int16)

# Process train and eval directories
root_file = '/home/ros_ws/dataset/audio_classes/test.npy'

# save file as test.wav
data = np.load(root_file).astype(np.uint8)
# data_converted = convert_uint8_to_int16(data)
data_converted = data.astype(np.uint8)
print(data, data.shape, data.dtype, data.max(), data.min()) 
print(data_converted, data_converted.shape, data_converted.dtype, data_converted.max(), data_converted.min())
write('/home/ros_ws/dataset/audio_classes/test.wav', 16000, data_converted)  # Assuming 16000 Hz sampling rate
