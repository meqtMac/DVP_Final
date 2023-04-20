import numpy as np
import scipy.special as sp
from typing import Callable
import Sources.Wave as Wave

def magnitude_ratio_squared_fast(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Compute the element-wise ratio of the squared magnitudes of two complex arrays.

    Args:
    - arr1 (np.ndarray): the first complex array.
    - arr2 (np.ndarray): the second complex array.

    Returns:
    - np.ndarray: a new 1D array of type float64 where each element is the squared ratio of
      the magnitude of arr1 to the magnitude of arr2 at the corresponding index.
    """
    # Compute the squared ratio of the magnitudes element-wise
    ratio_squared = np.square(np.abs(arr1 / arr2))

    return ratio_squared.astype(np.float64)

def filt(noisedSource: np.ndarray, noise: np.ndarray, filterGenerator: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Applies a given function to two arrays and returns an array.

    Args:
        x (np.ndarray): The first array.
        y (np.ndarray): The second array.
        func (callable): A function that takes two arrays as inputs and returns an array.

    Returns:
        np.ndarray: An array obtained by applying the given function to the two input arrays.
    """
    # validate the input arrays that they are both 1D arrays of the same length
    if noisedSource.ndim != 1 or noise.ndim != 1 or noisedSource.shape != noise.shape:
        raise ValueError("The input arrays must be 1D arrays of the same length.")
    
    Y = np.fft.fft(noisedSource)
    N = np.fft.fft(noise)

    gamma = magnitude_ratio_squared_fast(Y, N)
    H = filterGenerator(gamma, np.clip(gamma-1, 0, None))
    X = np.multiply(Y, H)
    return np.fft.ifft(X)


def placeholder_gain(gamma: np.ndarray, ksi: np.ndarray) -> np.ndarray:
    gain = np.ones(gamma.shape)
    return gain

# Spectral Subtraction
def spec_sub_gain(gamma: np.ndarray, ksi: np.ndarray) -> np.ndarray:
    gain = np.sqrt( ksi / (1+ ksi) ) # gain function
    return gain

# wiener 
def wiener_gain(gamma: np.ndarray, ksi: np.ndarray) -> np.ndarray:
    gain = ksi / (1+ ksi) # gain function
    return gain

# mmse
def mmse_stsa_gain(gamma: np.ndarray, ksi: np.ndarray) -> np.ndarray:
    vk = ksi 
    j0 = sp.jv(0, vk / 2)
    j1 = sp.jv(1, vk / 2)
        
    A = np.sqrt(np.pi * vk) / 2 / gamma
    B = (1 + vk) * j0 + vk * j1
    C = np.exp(-0.5 * vk)
    gain = A * B * C
    return gain

# log mmse
def logmmse_gain(gamma: np.ndarray, ksi: np.ndarray) -> np.ndarray:
    A = ksi / (1 + ksi)
    vk = A * gamma
    ei_vk = 0.5 * sp.expn(1, vk+1e-6)
    gain = A * np.exp(ei_vk, dtype=np.float64)
    return gain

# mmse sqr
def mmse_sqr_gain(gamma: np.ndarray, ksi: np.ndarray) -> np.ndarray:
		vk = ksi 
		A = ksi / (1 + ksi)
		B = (1 + vk) / gamma
		gain = np.sqrt(A*B)
		return gain

def map_joint_gain(gamma: np.ndarray, ksi: np.ndarray) -> np.ndarray:
	eps = 1e-6
	gain = (ksi + np.sqrt(ksi^ 2 + 2 * (1.0 + ksi)* ksi/ (gamma + eps))) / 2.0/ (1.0 + ksi)
	return gain

def process(noisedSource: Wave.FramedAudio, noise: Wave.FramedAudio, filterGenerator: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Apply a filter to each frame of a framed audio signal, using a noise signal as input for the filter.

    Args:
    - noisedSource (FramedAudio): a framed audio signal containing the noisy source signal.
    - noise (FramedAudio): a framed audio signal containing the noise signal.
    - filterGenerator (Callable[[np.ndarray, np.ndarray], np.ndarray]): a function that takes in two 1D numpy arrays (the noise and the noisy source), and returns a 1D numpy array (the filter).

    Returns:
    - filtered_audio (np.ndarray): a 1D numpy array containing the filtered audio signal.
    """
    filtered_frames = []
    for i in range(noisedSource.get_num_frames()):
        noisy_frame = noisedSource.get_frame(i)
        noise_frame = noise.get_frame(i)
        filter_frame = np.real(filt(noisy_frame, noise_frame, filterGenerator))
        filtered_frames.append(filter_frame)
    return np.concatenate(filtered_frames, dtype=np.float64)