import numpy as np
import scipy.io.wavfile as wavfile
from typing import Tuple, Union

class WavFile:
    """
    A class representing a .wav file.

    Attributes:
    - filename (str): the path to the .wav file.
    - sample_rate (int): the sample rate of the audio data in Hz.
    - audio_data (np.ndarray): the audio data as a float array between -1 and 1.

    Methods:
    - get_audio_data() -> np.ndarray: returns the audio data as a np.ndarray.
    - get_sample_rate() -> int: returns the sample rate.
    - save(filename: str = None) -> None: saves the audio data to a new .wav file.
    """

    ## add an init that initizal from a wave date and sample rate
    def __init__(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """
        Initialize the WavFile object.

        Args:
        - audio_data (np.ndarray): the audio data as a float array between -1 and 1.
        - sample_rate (int): the sample rate of the audio data in Hz.
        """
        self.sample_rate = sample_rate
        self.audio_data = audio_data

    @classmethod
    def fromfile(cls, filename: str) -> None:
        """
        Initialize the WavFile object.

        Args:
        - filename (str): the path to the .wav file.
        """
        # self.filename = filename
        sample_rate: int
        audio_data: np.ndarray
        sample_rate, audio_data = wavfile.read(filename)
        audio_data = audio_data.astype(np.float32) / 32767.0
        return cls(audio_data, sample_rate)

    def get_audio_data(self) -> np.ndarray:
        """
        Returns the audio data as a np.ndarray.

        Returns:
        - np.ndarray: the audio data as a np.ndarray.
        """
        return self.audio_data

    def get_sample_rate(self) -> int:
        """
        Returns the sample rate.

        Returns:
        - int: the sample rate.
        """
        return self.sample_rate

    # add a function that return a FramedAudio object
    def get_framed_audio(self, frame_size: int, hop_size: int) -> "FramedAudio":
        """
        Returns a FramedAudio object.

        Args:
        - frame_size (int): the size of each audio frame in samples.
        - hop_size (int): the number of samples between the start of each audio frame.

        Returns:
        - FramedAudio: a FramedAudio object.
        """
        return FramedAudio(self.audio_data, self.sample_rate, frame_size, hop_size)
    
    def save(self, filename: Union[str, None] = None ) -> None:
        """
        Saves the audio data to a new .wav file.

        Args:
        - filename (str): the path to the new .wav file. If None, the original filename will be used.
        """
        if filename is None:
            filename = self.filename
        audio_data_int16 = (self.audio_data * 32767.0).astype(np.int16)
        wavfile.write(filename, self.sample_rate, audio_data_int16)

    # add a noise generator that return a Wave object, which return a noise with the same sample rate and length of the original wave
    def generate_noise(self, noise_type: str = "white", noise_level: float = 0.1) -> "WavFile":
        """
        Generates a noise signal.

        Args:
        - noise_type (str): the type of noise. Can be "white" or "pink".
        - noise_level (float): the level of the noise. A value between 0 and 1.

        Returns:
        - WavFile: a WavFile object containing the noise signal.
        """
        if noise_type == "white":
            noise = np.random.normal(0.0, 1.0, self.audio_data.shape)
        elif noise_type == "pink":
            noise = np.random.normal(0.0, 1.0, self.audio_data.shape)
            for i in range(1, len(noise)):
                noise[i] += 0.5 * noise[i - 1]
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        noise = noise / np.max(np.abs(noise))
        noise = noise * noise_level
        return WavFile(noise, self.sample_rate)

    # add a function that add a noise which is a WaveFile object to the original wave
    def add_noise(self, noise: "WavFile") -> None:
        """
        Adds a noise signal to the audio data.

        Args:
        - noise (WavFile): a WavFile object containing the noise signal.
        - noise_level (float): the level of the noise. A value between 0 and 1.
        """
        if noise.sample_rate != self.sample_rate:
            raise ValueError("The noise signal must have the same sample rate as the audio data.")
        if len(noise.audio_data) != len(self.audio_data):
            raise ValueError("The noise signal must have the same length as the audio data.")
        self.audio_data += noise.audio_data 

    # add a funtion to plot the Wave, with x axis in seconds
    def plot(self, 
             start_time: float = 0.0, 
             end_time: float = None, 
             title: str = None, 
             xlabel: str = "t/s", 
             ylabel: str = None) -> None:
        """
        Plots the audio data.

        Args:
        - start_time (float): the start time of the plot in seconds.
        - end_time (float): the end time of the plot in seconds. If None, the end of the audio data will be used.
        - title (str): the title of the plot.
        - xlabel (str): the label of the x-axis.
        - ylabel (str): the label of the y-axis.
        """
        import matplotlib.pyplot as plt
        start_index = int(start_time * self.sample_rate)
        if end_time is None:
            end_index = len(self.audio_data)
        else:
            end_index = int(end_time * self.sample_rate)
        plt.plot(np.arange(start_index, end_index) / self.sample_rate, self.audio_data[start_index:end_index])
        if title is not None:
            plt.title(title)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.show()


# funciton that add two wave files together and return a new wave file
def add_waves(wave1: WavFile, wave2: WavFile) -> WavFile:
    """
    Adds two WavFile objects together.

    Args:
    - wave1 (WavFile): the first WavFile object.
    - wave2 (WavFile): the second WavFile object.

    Returns:
    - WavFile: a WavFile object containing the sum of the two input WavFile objects.
    """
    if wave1.sample_rate != wave2.sample_rate:
        raise ValueError("The two WavFile objects must have the same sample rate.")
    if len(wave1.audio_data) != len(wave2.audio_data):
        raise ValueError("The two WavFile objects must have the same length.")
    return WavFile(wave1.audio_data + wave2.audio_data, wave1.sample_rate)

class FramedAudio:
    """
    A class representing a framed audio signal.

    Attributes:
    - audio_data (np.ndarray): the audio data as a 1D np.ndarray.
    - sample_rate (int): the sample rate of the audio signal in Hz.
    - frame_size (int): the size of each audio frame in samples.
    - hop_size (int): the number of samples between the start of each audio frame.
    - num_frames (int): the total number of audio frames.
    - frames (List[np.ndarray]): a list of audio frames, each represented as a 1D np.ndarray.

    Methods:
    - get_frame(index: int) -> np.ndarray: returns a specific audio frame as a 1D np.ndarray.
    - get_frame_time(index: int) -> float: returns the start time of a specific audio frame in seconds.
    - get_num_frames() -> int: returns the total number of audio frames.
    """

    def __init__(self, audio_data: np.ndarray, sample_rate: int, frame_size: int, hop_size: int) -> None:
        """
        Initialize the FramedAudio object.

        Args:
        - audio_data (np.ndarray): the audio data as a 1D np.ndarray.
        - sample_rate (int): the sample rate of the audio signal in Hz.
        - frame_size (int): the size of each audio frame in samples.
        - hop_size (int): the number of samples between the start of each audio frame.
        """
        self.audio_data: np.ndarray = audio_data
        self.sample_rate: int = sample_rate
        self.frame_size: int = frame_size
        self.hop_size: int = hop_size
        self.num_frames: int = int(np.ceil((len(self.audio_data) - self.frame_size) / self.hop_size)) + 1
        self.frames = []
        for i in range(self.num_frames):
            start = i * self.hop_size
            end = min(start + self.frame_size, len(self.audio_data))
            frame = np.zeros(self.frame_size)
            frame[:end - start] = self.audio_data[start:end]
            self.frames.append(frame)

    def get_frame(self, index: int) -> np.ndarray:
        """
        Returns a specific audio frame as a 1D np.ndarray.

        Args:
        - index (int): the index of the audio frame.

        Returns:
        - np.ndarray: the audio frame as a 1D np.ndarray.
        """
        return self.frames[index]

    def get_frame_time(self, index: int) -> float:
        """
        Returns the start time of a specific audio frame in seconds.

        Args:
        - index (int): the index of the audio frame.

        Returns:
        - float: the start time of the audio frame in seconds.
        """
        return index * self.hop_size / self.sample_rate

    def get_num_frames(self) -> int:
        """
        Returns the total number of audio frames.

        Returns:
        - int: the total number of audio frames.
        """
        return self.num_frames

    # add a function that return a WavFile object
    def get_wav_file(self) -> WavFile:
        """
        Returns a WavFile object.

        Returns:
        - WavFile: a WavFile object.
        """
        return WavFile(self.audio_data, self.sample_rate)
    
    # return the spectrogram of the framed audio
    def get_spectrogram(self) -> np.ndarray:
        """
        Returns the spectrogram of the framed audio.

        Returns:
        - np.ndarray: the spectrogram of the framed audio.
        """
        return np.abs(np.fft.rfft(self.frames, axis=1))

    # plot the spectrogram of the framed audio, with x showing the time of frame and y showing the frequency
    def plot_spectrogram(self, 
                         start_time: float = 0.0, 
                         end_time: float = None, 
                         title: str = None, 
                         xlabel: str = "t/s", 
                         ylabel: str = "f/Hz") -> None:
        """
        Plots the spectrogram of the framed audio.

        Args:
        - start_time (float): the start time of the plot in seconds.
        - end_time (float): the end time of the plot in seconds. If None, the end of the audio data will be used.
        - title (str): the title of the plot.
        - xlabel (str): the label of the x-axis.
        - ylabel (str): the label of the y-axis.
        """

        import matplotlib.pyplot as plt
        # Compute the spectrogram using a short-time Fourier transform
        spec, freq, _, _ = plt.specgram(self.audio_data, 
                                        NFFT=self.frame_size, 
                                        Fs=self.sample_rate, 
                                        cmap='plasma', 
                                        noverlap=self.frame_size - self.hop_size, mode='magnitude'
                                        )

        # Convert the spectrogram to dB
        spec_db = 20 * np.log10(spec)
        
        # Plot the spectrogram
        plt.imshow(spec_db, origin='lower', aspect='auto', cmap='plasma', extent=[0, self.get_frame_time(self.get_num_frames() - 1), freq[0], freq[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        # add and describe for the colorbar
        cbar = plt.colorbar() 
        cbar.set_label('Magnitude (dB)')
        plt.show()