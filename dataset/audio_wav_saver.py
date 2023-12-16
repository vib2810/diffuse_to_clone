import os
import numpy as np
from scipy.io.wavfile import write
count = 0

def convert_uint8_to_int16(data):
    # Convert from uint8 to float32, then scale to int16 range and convert to int16
    data = data.astype(np.float32)
    data = (data - 128) / 128  # Normalize to -1 to 1
    data = data * 32767        # Scale to int16 range
    return data.astype(np.int16)

def process_directory(input_dir, output_dir):
    global count
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all subdirectories
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)

        if os.path.isdir(subdir_path):
            output_subdir = os.path.join(output_dir, subdir)

            # Create corresponding subdirectory in output directory
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            # Process each npz file in the subdirectory
            for file in os.listdir(subdir_path):
                if file.endswith('.npy'):
                    file_path = os.path.join(subdir_path, file)
                    output_file_path = os.path.join(output_subdir, os.path.splitext(file)[0] + '.wav')

                    # Load audio data from npy file
                    data = np.load(file_path)
                    data_converted = convert_uint8_to_int16(data)
                    if count == 0:
                        print(data, data.shape, data.dtype, data.max(), data.min())
                        print(data_converted, data_converted.shape, data_converted.dtype, data_converted.max(), data_converted.min())
                        count += 1
                    

                    # Save audio data as wav file
                    write(output_file_path, 16000, data_converted)  # Assuming 16000 Hz sampling rate


# Process train and eval directories
root = '/home/ros_ws/dataset/audio_classes'
process_directory(root+'/train', root+'/train_wav')
process_directory(root+'/eval', root+'/eval_wav')
