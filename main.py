import numpy as np
import scipy.io.wavfile as wavfile
import Sources.Wave as wave
import Sources.DSP as dsp
import matplotlib.pyplot as plt

# create of fig of 4*2 subplots
fig, axs = plt.subplots(4, 4, figsize=(10, 5))

# fig = plt.figure()
audio = wave.WavFile.fromfile('Resources/Audio/beijing.wav')
noise = audio.generate_noise(noise_level=0.5)
audioAddNoise = wave.add_waves(audio, noise)

audio.plot(fig=fig, ax=axs[0][0])
noise.plot(fig=fig,ax=axs[0][1])
audioAddNoise.plot(fig=fig, ax=axs[0][2])

framed1 = audio.get_framed_audio(800, 800)
framed1.plot_spectrogram(fig=fig, ax=axs[1][0])

framed2 = noise.get_framed_audio(800, 800)
framed2.plot_spectrogram(fig=fig, ax=axs[1][1])

framed3 = audioAddNoise.get_framed_audio(800, 800)
framed3.plot_spectrogram(fig=fig, ax=axs[1][2])

generated = dsp.process(framed3, framed2, dsp.mmse_sqr_gain)
generatedWave =  wave.WavFile(generated, audio.get_sample_rate())
generatedWave.plot(fig=fig, ax=axs[0][3])

generatedFramee = generatedWave.get_framed_audio(800, 800)
generatedFramee.plot_spectrogram(fig=fig, ax=axs[1][3])

processes = [dsp.spec_sub_gain, dsp.wiener_gain,  dsp.logmmse_gain, dsp.mmse_sqr_gain]
for i, process in enumerate(processes):
    generated = dsp.process(framed3, framed2, process)
    generatedWave =  wave.WavFile(generated, audio.get_sample_rate())
    generatedWave.plot(fig=fig, ax=axs[2][i])
    generatedFramee = generatedWave.get_framed_audio(800, 800)
    generatedFramee.plot_spectrogram(fig=fig, ax=axs[3][i])

plt.tight_layout()
# plt.show()
plt.savefig("test.pdf")