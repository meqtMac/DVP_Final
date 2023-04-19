import numpy as np
import scipy.io.wavfile as wavfile
import Sources.Wave as wave

audio = wave.WavFile.fromfile('Resources/Audio/beijing.wav')
noise = audio.generate_noise()
audioAddNoise = wave.add_waves(audio, noise)

framed1 = audio.get_framed_audio(800, 100)
framed1.plot_spectrogram()

framed2 = noise.get_framed_audio(800, 100)
framed2.plot_spectrogram()

framed3 = audioAddNoise.get_framed_audio(800, 100)
framed3.pl
