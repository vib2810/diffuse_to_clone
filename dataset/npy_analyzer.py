import matplotlib.pyplot as plt
from pydub import AudioSegment
from ros import rosbag
import numpy as np
from io import BytesIO
import os
import preprocess_audio

def create_audio_amp_from_numpy_array(np_array_path, sample_rate=16000, sample_width=2, channels=1):
    """ 
    Does not work for some reason
    """
    # Load the NumPy array and ensure it matches the audio format (16-bit PCM in this case)
    np_array = np.load(np_array_path).astype(np.uint8)

    # Convert the NumPy array to bytes
    audio_data_bytes = np_array.tobytes()

    # Create an audio segment using from_raw with the correct parameters
    audio_segment = AudioSegment.from_raw(
        BytesIO(audio_data_bytes),
        sample_width=sample_width,
        frame_rate=sample_rate,
        channels=channels,
        format="raw"  # Specify the format as raw
    )

    return np.array(audio_segment.get_array_of_samples())

def get_audio_amplitude_array(npy_file_path):
    """
    Save npy as mp3 if not already saved
    Load mp3 as audio segment
    Return audio segment amplitudes as numpy array
    """
    loaded_audio_data = np.load(npy_file_path).astype(np.int8)

    # assign the corresponding mp3 file path
    mp3_path = npy_file_path[:-4] + '.mp3'
    
    # check if mp3 file exists
    if not os.path.exists(mp3_path):
        # Assuming the loaded data is in the correct format for MP3
        with open(mp3_path, 'wb') as mp3_file:
            mp3_file.write(loaded_audio_data.tobytes())

    audio_segment = AudioSegment.from_mp3(mp3_path)
    return np.array(audio_segment.get_array_of_samples())
    
def plot_amplitude(audio_amps):
    # print shape
    print('Shape of samples: {}'.format(audio_amps.shape))

    plt.figure(figsize=(20, 4))
    plt.plot(audio_amps)
    plt.title('Amplitude Signal')
    plt.ylabel('Amplitude')
    plt.xlabel('Sample')
    plt.show()


numpy_path = '/home/ros_ws/dataset/audio_classes/train/2/21.npy'
# audio_amps = get_audio_amplitude_array(numpy_path)
# plot_amplitude(audio_amps)

preprocessed_audio = preprocess_audio.process_audio(numpy_path, sample_rate=16000, check_valid=True)