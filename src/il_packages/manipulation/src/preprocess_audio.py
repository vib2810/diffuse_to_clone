import glob
import os
import math
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Ignore frequency components below this value (in Hz)
MIN_RELEVANT_FREQUENCY = 0
# Ignore frequency components above this value (in Hz)
MAX_RELEVANT_FREQUENCY = 12500

def process_audio(audio_data, sample_rate=16000, num_freq_bins=100, num_time_bins=57, check_valid=True):
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
  fully_binned_spectrogram, binned_freq_spectrogram = compute_spectrogram(audio_data, sample_rate, num_freq_bins, num_time_bins)

  # This is for debugging any invalid spectrograms that slip through the cracks.
  if check_valid and np.mean(fully_binned_spectrogram) < 1:
    print(np.mean(fully_binned_spectrogram))
    plt.imshow(binned_freq_spectrogram)
    plt.colorbar()
    plt.show()
    plt.imshow(fully_binned_spectrogram)
    plt.colorbar()
    plt.show()
  
  # fully_binned_spectrogram = (fully_binned_spectrogram - np.min(fully_binned_spectrogram))/(np.max(fully_binned_spectrogram) - np.min(fully_binned_spectrogram))
  fully_binned_spectrogram = fully_binned_spectrogram/300
  
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
  # print(f"Shape of Sxx: {Sxx.shape}")
  print(f"Min value: {np.min(Sxx)}, max value: {np.max(Sxx)}")

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