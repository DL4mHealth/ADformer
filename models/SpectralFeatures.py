import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.signal
import scipy.fftpack
from scipy.signal import coherence, welch, hilbert


def spectral_feature_extractor(x_enc, sampling_rate):
    """
    Extract spectral features from a frequency band of a batch of trials.
    Map batch with shape BxTxC to BxFxCx.
    B is the batch size, T is timestamps, C is channels, F is spectral features.

    Spectral features names: Phase Shift, Phase Coherence, Coherence, Bispectrum, Bicoherence,
    Spectral Centroid, Spectral Roll-off, Spectral Peak, Average Magnitude, Power Spectrum Density,
    Median Frequency, Amplitude Modulation

    Parameters:
      x_enc : Tensor
        A batch of samples with shape BxTxC
      sampling_rate : float
        Sampling rate of the input signals

    Returns:
      output : Tensor
        A batch of features with shape BxFxC
    """

    def compute_phase_shift(x):
        return np.angle(np.fft.fft(x, axis=1))

    def compute_phase_coherence(x):
        phase = np.angle(np.fft.fft(x, axis=1))
        return np.abs(np.mean(np.exp(1j * phase), axis=1))

    def compute_bispectrum(x):
        return np.abs(np.fft.fft(x, axis=1))

    def compute_bicoherence(x):
        bispec = compute_bispectrum(x)
        return bispec / np.sqrt(np.sum(bispec ** 2, axis=1, keepdims=True))

    def compute_spectral_centroid(x, fs):
        magnitude_spectrum = np.abs(np.fft.fft(x, axis=1))
        freqs = np.fft.fftfreq(x.shape[1], 1 / fs)
        freqs = freqs[:, np.newaxis]
        return np.sum(freqs * magnitude_spectrum, axis=1) / np.sum(magnitude_spectrum, axis=1)

    def compute_spectral_rolloff(x, fs, rolloff=0.85):
        magnitude_spectrum = np.abs(np.fft.fft(x, axis=1))
        cumulative_sum = np.cumsum(magnitude_spectrum, axis=1)
        threshold = rolloff * cumulative_sum[:, -1]
        return np.argmax(cumulative_sum >= threshold[:, None], axis=1)

    def compute_spectral_peak(x):
        magnitude_spectrum = np.abs(np.fft.fft(x, axis=1))
        return np.max(magnitude_spectrum, axis=1)

    def compute_average_magnitude(x):
        magnitude_spectrum = np.abs(np.fft.fft(x, axis=1))
        return np.mean(magnitude_spectrum, axis=1)

    def compute_power_spectrum_density(x, fs):
        f, Pxx = welch(x, fs=fs, axis=1)
        return Pxx

    def compute_median_frequency(x, fs):
        magnitude_spectrum = np.abs(np.fft.fft(x, axis=1))
        cumulative_sum = np.cumsum(magnitude_spectrum, axis=1)
        half_total_energy = cumulative_sum[:, -1] / 2

        median_freq_idx = np.zeros((x.shape[0], x.shape[2]), dtype=int)
        for i in range(x.shape[0]):
            for j in range(x.shape[2]):
                median_freq_idx[i, j] = np.where(cumulative_sum[i, :, j] >= half_total_energy[i, j])[0][0]

        freqs = np.fft.fftfreq(x.shape[1], 1 / fs)
        return freqs[median_freq_idx]

    def compute_amplitude_modulation(x):
        hilbert_transform = np.abs(hilbert(x, axis=1))
        return np.std(hilbert_transform, axis=1)

    x_enc_np = x_enc.cpu().numpy()
    B, T, C = x_enc_np.shape

    phase_shift = compute_phase_shift(x_enc_np)
    phase_coherence = compute_phase_coherence(x_enc_np)
    bispectrum = compute_bispectrum(x_enc_np)
    bicoherence = compute_bicoherence(x_enc_np)
    spectral_centroid = compute_spectral_centroid(x_enc_np, sampling_rate)
    spectral_rolloff = compute_spectral_rolloff(x_enc_np, sampling_rate)
    spectral_peak = compute_spectral_peak(x_enc_np)
    average_magnitude = compute_average_magnitude(x_enc_np)
    power_spectrum_density = compute_power_spectrum_density(x_enc_np, sampling_rate)
    median_frequency = compute_median_frequency(x_enc_np, sampling_rate)
    amplitude_modulation = compute_amplitude_modulation(x_enc_np)

    features = np.concatenate([
        phase_shift,
        phase_coherence[:, np.newaxis, :],
        bispectrum,
        bicoherence,
        spectral_centroid[:, np.newaxis, :],
        spectral_rolloff[:, np.newaxis, :],
        spectral_peak[:, np.newaxis, :],
        average_magnitude[:, np.newaxis, :],
        power_spectrum_density,
        median_frequency[:, np.newaxis, :],
        amplitude_modulation[:, np.newaxis, :]
    ], axis=1)

    output = torch.tensor(features, dtype=torch.float32).to(x_enc.device)
    return output


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.sampling_rate = configs.sampling_rate

        self.encoder = spectral_feature_extractor

        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            raise NotImplementedError
        if self.task_name == 'imputation':
            raise NotImplementedError
        if self.task_name == 'anomaly_detection':
            raise NotImplementedError
        if self.task_name == 'classification':
            self.projection = nn.Linear(configs.enc_in *
                                        (self.seq_len * 3 + (math.floor(self.seq_len/2) + 1) + 7), configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raise NotImplementedError

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError

    def anomaly_detection(self, x_enc):
        raise NotImplementedError

    def classification(self, x_enc, x_mark_enc):  # (batch_size, timestamps, enc_in)
        enc_out = self.encoder(x_enc, self.sampling_rate)  # (batch_size, features, enc_in)
        enc_out = enc_out.reshape(enc_out.shape[0], -1)  # (batch_size, features * enc_in)

        output = self.projection(enc_out)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            raise NotImplementedError
        if self.task_name == 'imputation':
            raise NotImplementedError
        if self.task_name == 'anomaly_detection':
            raise NotImplementedError
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None