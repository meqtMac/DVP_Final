import numpy as np
import scipy.io.wavfile as wavfile
import Sources.Wave as wave
import Sources.DSP as dsp
import matplotlib.pyplot as plt

# create of fig of 4*2 subplots
fig, axs = plt.subplots(1, 1, figsize=(10, 2.5))

# fig = plt.figure()
audio = wave.WavFile.fromfile('Resources/Audio/beijing.wav')
noise = audio.generate_noise(noise_level=0.1)
# noise.save('Resources/Audio/noise.wav')

audioAddNoise = wave.add_waves(audio, noise)
# audioAddNoise.save('Resources/Audio/audioAddNoise.wav')

fig, axs = plt.subplots(1, 1, figsize=(10, 2.5))
audio.plot(fig=fig, ax=axs)
plt.savefig("Results/wave_audio.jpg")

fig, axs = plt.subplots(1, 1, figsize=(10, 2.5))
noise.plot(fig=fig,ax=axs)
plt.savefig("Results/wave_noise.jpg")

framed1 = audio.get_framed_audio(800, 800)

fig, axs = plt.subplots(1, 1, figsize=(10, 2.5))
framed1.plot_spectrogram(fig=fig, ax=axs)
plt.savefig("Results/spectrogram_audio.jpg")

framed2 = noise.get_framed_audio(800, 800)

fig, axs = plt.subplots(1, 1, figsize=(10, 2.5))
framed2.plot_spectrogram(fig=fig, ax=axs)
plt.savefig("Results/spectrogram_noise.jpg")

framed3 = audioAddNoise.get_framed_audio(800, 800)

processes = [dsp.placeholder_gain ,
             dsp.spec_sub_gain, 
             dsp.wiener_gain,  
             dsp.mmse_stsa_gain,
             dsp.logmmse_gain, 
             dsp.mmse_sqr_gain,
             ]

error_lambda = [wave.snr, 
                wave.mse, 
                wave.msle, 
                wave.mae, 
                ]

print("SNR/dB\tMSE\tMSLE\tMAE")
for i, process in enumerate(processes):

    generated = dsp.process(framed3, framed2, process)
    generatedWave =  wave.WavFile(generated, audio.get_sample_rate())
    # generatedWave.save('Resources/Audio/generated'+str(i)+'.wav')
    audio_processed = dsp.process(framed1, framed2, process)
    audio_wave = wave.WavFile(audio_processed, audio.get_sample_rate())
 
    fig, axs = plt.subplots(1, 1, figsize=(10, 2.5))
    generatedWave.plot(fig=fig, ax=axs)
    plt.savefig("Results/wave"+str(i)+".jpg")

    generatedFramee = generatedWave.get_framed_audio(800, 800)

    fig, axs = plt.subplots(1, 1, figsize=(10, 2.5)) 
    generatedFramee.plot_spectrogram(fig=fig, ax=axs)
    plt.savefig("Results/spectrogram"+str(i)+".jpg")

    # print a table row for a list of errors, with the first parameter passed in as generatedWave, and the second as audio
    print(f"{error_lambda[0](generatedWave, audio_wave):.8f}\t{error_lambda[1](generatedWave, audio_wave):.8f}\t{error_lambda[2](generatedWave, audio_wave):.8f}\t{error_lambda[3](generatedWave, audio_wave):.8f}")

# plt.tight_layout()
# plt.show()
# plt.savefig("test.pdf")