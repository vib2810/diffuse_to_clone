import glob
import os
import math
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from pydub import AudioSegment
import pickle
from scipy.ndimage import zoom
import cv2

# Ignore frequency components below this value (in Hz)
MIN_RELEVANT_FREQUENCY = 0
# Ignore frequency components above this value (in Hz)
MAX_RELEVANT_FREQUENCY = 12500

def get_audio_amplitude_array(npy_file_path):
    """
    Save npy as mp3 if not already saved
    Load mp3 as audio segment
    Return audio segment amplitudes as numpy array
    """
    loaded_audio_data = np.load(npy_file_path).astype(np.int8)

    # store amp as pickle
    audio_amp_path = npy_file_path[:-4] + '_amp.pkl'
    
    # check if audio amp array exists
    if not os.path.exists(audio_amp_path):      
      mp3_path = npy_file_path[:-4] + '.mp3'
      # Assuming the loaded data is in the correct format for MP3
      with open(mp3_path, 'wb') as mp3_file:
          mp3_file.write(loaded_audio_data.tobytes())
          
      # write audio amp array
      amp_array = np.array(AudioSegment.from_mp3(mp3_path).get_array_of_samples(), dtype=np.int16)
      pickle.dump(amp_array, open(audio_amp_path, 'wb'))
    
    else:
      amp_array = pickle.load(open(audio_amp_path, 'rb'))
      amp_array =  np.array(amp_array, dtype=np.int16)

    return amp_array

def process_audio(audio_data_npy_path, sample_rate=16000, num_freq_bins=100, num_time_bins=57, check_valid=False):
  '''Computes and processes a binned spectrogram from a raw audio (unclipped and unpadded) signal array.

  Args:
    audio_data (numpy.array): Array for a raw audio signal (one channel only)
    sample_rate (int) : The number of samples per second of the audio signal.
    num_freq_bins (int) : The number of desired frequency bins in the processed binned spectrogram
    num_time_bins (int) : The number of desired time bins in the processed binned spectrogram
    check_valid (boolean) : Whether to interrupt the function on a processing error and debug with plots

  Returns:
    A numpy.array representing the processed and binned spectrogram
  '''
  audio_data = get_audio_amplitude_array(audio_data_npy_path) # shape (37440,)
  
  fully_binned_spectrogram = compute_spectrogram(audio_data, sample_rate, num_freq_bins, num_time_bins)
  fully_binned_spectrogram = fully_binned_spectrogram/(60000*50)

  # This is for debugging any invalid spectrograms that slip through the cracks.
  if check_valid:
    print("Fully Binned Spectrogram Min: ", np.min(fully_binned_spectrogram), " Max: ", np.max(fully_binned_spectrogram), " Mean: ", np.mean(fully_binned_spectrogram))
    # plt.imshow(binned_freq_spectrogram)
    # plt.colorbar()
    # plt.show()
    plt.figure()
    plt.plot(audio_data)
    plt.xlabel('Time (samples)'); plt.ylabel('Amplitude')
    plt.figure()
    plt.imshow(fully_binned_spectrogram)
    plt.ylabel('Time (bins)'); plt.xlabel('Frequency (bins)')
    plt.colorbar()
    plt.show()
  
  return fully_binned_spectrogram

def compute_spectrogram(audio_data, sample_rate, num_freq_bins, num_time_bins):
  '''Computes and processes a spectrogram directly from an audio signal.

  Args:
    audio_data (numpy.array): Array for a raw audio signal (one channel only)
    sample_rate (int) : The number of samples per second of the audio signal.
    num_freq_bins (int) : The number of desired frequency bins in the processed binned spectrogram
    num_time_bins (int) : The number of desired time bins in the processed binned spectrogram

  Returns:
    A numpy.array representing the fully processed and binned spectrogram
    A numpy.array for the processed spectrogram only binned on the frequency dimension (for debugging purposes)
  '''
  # Sxx has first dim Freq, second dim time
  f, t, Sxx = signal.spectrogram(audio_data, sample_rate, scaling='spectrum', return_onesided=True)
  Sxx = np.array(Sxx) # shape (129, 167), f shape (129,)
  plt.imshow(Sxx)
  
  assert Sxx.shape[0]/num_freq_bins > 1, f"num_freq_bins {num_freq_bins} is more than Sxx.shape[0] {Sxx.shape[0]}"
  assert Sxx.shape[1]/num_time_bins > 1, f"num_time_bins {num_time_bins} is more than Sxx.shape[1] {Sxx.shape[1]}"

  fully_binned_spectrogram = bin_matrix(Sxx, (num_freq_bins, num_time_bins))
  return fully_binned_spectrogram.T
  
def bin_matrix(original_matrix, new_shape):
    """
    Bins a matrix to a new shape using bilinear interpolation with cv2.

    Args:
        original_matrix (numpy.array): The original matrix to be binned.
        new_shape (tuple): The shape of the new binned matrix (new_height, new_width).

    Returns:
        numpy.array: The binned matrix.
    """
    # OpenCV's resize function takes size in (width, height) order
    size = (new_shape[1], new_shape[0])

    # Use cv2's resize function with bilinear interpolation
    binned_matrix = cv2.resize(original_matrix, size, interpolation=cv2.INTER_LINEAR)

    return binned_matrix