import glob
import os
import math
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from pydub import AudioSegment
import pickle

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
  
  fully_binned_spectrogram, binned_freq_spectrogram = compute_spectrogram(audio_data, sample_rate, num_freq_bins, num_time_bins)
  fully_binned_spectrogram = fully_binned_spectrogram/60000

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
  Sxx = np.array(Sxx)
  
  assert Sxx.shape[0]/num_freq_bins > 1, f"num_freq_bins {num_freq_bins} is more than Sxx.shape[0] {Sxx.shape[0]}"
  assert Sxx.shape[1]/num_time_bins > 1, f"num_time_bins {num_time_bins} is more than Sxx.shape[1] {Sxx.shape[1]}"
  
  # print(f"Shape of Sxx: {Sxx.shape}")
  # print(f"Min value: {np.min(Sxx)}, max value: {np.max(Sxx)}")

  # plot spectrogram Sxx
  # plt.pcolormesh(t, f, Sxx)
  # plt.ylabel('Frequency [Hz]')
  # plt.xlabel('Time [sec]')
  # plt.show(block=False)
  # plt.pause(0.0001)

  # Find the indices of the bounds of the relevant frequencies
  min_relevant_freq_idx = np.searchsorted(f, MIN_RELEVANT_FREQUENCY)
  max_relevant_freq_idx = np.searchsorted(f, MAX_RELEVANT_FREQUENCY)

  trimmed_spectrogram = Sxx[min_relevant_freq_idx:max_relevant_freq_idx,:]
  trimmed_freqs = f[min_relevant_freq_idx:max_relevant_freq_idx]

  binned_freq_spectrogram = bin_spectrogram_freq(trimmed_spectrogram, num_freq_bins)
  fully_binned_spectrogram = bin_spectrogram_time(binned_freq_spectrogram, num_time_bins)

  return fully_binned_spectrogram, binned_freq_spectrogram

def bin_spectrogram_freq(spectrogram, num_freq_bins):
  '''Bins a spectrogram on its frequency dimension.

  Args:
    spectrogram (numpy.array) : The unbinned spectrogram
    num_freq_bins (int) : The number of desired frequency bins in the processed binned spectrogram

  Returns:
    The spectrogram binned on its frequency dimension.
  '''
  return __bin_matrix_dimension(spectrogram, 0, num_freq_bins)

def bin_spectrogram_time(spectrogram, num_time_bins):
  '''Bins a spectrogram on its time dimension.

  Args:
    spectrogram (numpy.array) : The unbinned spectrogram
    num_time_bins (int) : The number of desired time bins in the processed binned spectrogram

  Returns:
    The spectrogram binned on its time dimension.
  '''
  return __bin_matrix_dimension(spectrogram, 1, num_time_bins)

def __bin_matrix_dimension(m, dimension, num_bins):
  '''Bins a matrix on a specified dimension.

  Args:
    m (numpy.array) : The original matrix
    dimension (int) : The dimension to bin
    num_bins (int) : The desired number of bins for the specified dimension

  Returns:
    A numpy.array of the matrix binned on the specified dimension.
  '''
  # print(f"Shape of m: {m.shape}")
  bin_size = int(np.floor(m.shape[dimension]/(num_bins+0.0)))

  binned_matrix = np.zeros((m.shape[1-dimension], num_bins))
  
  for b in range(num_bins):
    min_bin_idx = b * bin_size
    max_bin_idx = min((b+1) * bin_size, m.shape[dimension])
    if dimension == 0:
        binned_matrix[:,b] = np.sum(m[min_bin_idx:max_bin_idx, :], axis=0)
    else:
        binned_matrix[:,b] = np.sum(m[:, min_bin_idx:max_bin_idx], axis=1)

  # print(f"Shape of binned_matrix: {binned_matrix.shape}")
  return binned_matrix.T