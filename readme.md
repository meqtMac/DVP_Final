# Digital Voice Processing Final Project

## Code Structure
- WaveFile: a class to read and write wave files, store audio metadata, and export frames of audio data for processing and do time domain, and spectrum domain time.
- FramedAudio: a class to store frames of audio data and perform operations on them
- filterdesign: function that generate frequency response from Prior SNR, and Posterior SNR
    - dsp.spec_sub_gain: Spectrum Subtraction
    - dsp.wiener_gain: Wiener Filter 
    - dsp.mmse_stsa_gain: Bayesian MMSE
    - dsp.logmmse_gain: Bayesian LogMMSE
    - dsp.mmse_sqr_gain: Bayesian MMSE Square
- utilsfunctions:
    - add_noise: generate noise
    - measure:
        - wave.snr: SNR
        - wave.mse: Mean Square Error
        - wave.msle: Mean Square Log Error
        - wave.mae: Mean Absolute Error


