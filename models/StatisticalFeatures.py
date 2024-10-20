import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def bandpass_filter(signal, fs, lowcut, highcut):
    # length of signal
    fft_len = signal.size(1)
    # FFT
    fft_spectrum = torch.fft.rfft(signal, n=fft_len, dim=1)
    # get frequency bins
    freqs = torch.fft.rfftfreq(fft_len, d=1/fs)
    # create mask for freqs
    mask = (freqs >= lowcut) & (freqs <= highcut)
    # expand mask to match fft_spectrum
    mask = mask.view(1, -1, 1).expand_as(fft_spectrum)
    mask = mask.to(signal.device)
    # apply mask
    fft_spectrum = fft_spectrum * mask
    # IFFT
    filtered_signal = torch.fft.irfft(fft_spectrum, n=fft_len, dim=1)

    return filtered_signal


def extract_bands(signal, fs):
    # Signal bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 100)
    }

    # Filter signals within specified frequency bands and concatenate on channel axis
    band_signals = [bandpass_filter(signal, fs, low, high) for low, high in bands.values()]
    return torch.cat(band_signals, dim=-1)


def statistical_feature_extractor(x_enc):
    """
    Extract features from a frequency band of a batch of trials. Map batch with shape BxTxC to Bx(F*C).
    B is the batch size, T is timestamps, C is channels, F is statistical features.
    Statistical features names: Mean, Variance, Skewness, Kurtosis,
    Standard Deviation, Interquartile Range, Maximum, Minimum, Average, Median

    Parameters:
      x_enc : Tensor
        A batch of samples with shape BxTxC

    Returns:
      output : Tensor
        A batch of features with shape Bx(F*C)
    """
    epsilon = 1e-9  # Small constant to prevent division by zero or log of zero

    # Compute basic statistics
    mean = torch.mean(x_enc, dim=1)
    min = torch.min(x_enc, dim=1).values
    max = torch.max(x_enc, dim=1).values
    std = torch.std(x_enc, dim=1, unbiased=False)  # Set unbiased=False for population std
    var = torch.var(x_enc, dim=1, unbiased=False)  # Set unbiased=False for population variance
    median = torch.median(x_enc, dim=1).values

    # Compute IQR
    q75 = torch.quantile(x_enc, 0.75, dim=1, keepdim=True)
    q25 = torch.quantile(x_enc, 0.25, dim=1, keepdim=True)
    iqr = (q75 - q25).squeeze(1)

    # Calculate the 3rd and 4th moments for skewness and kurtosis
    deviations = x_enc - mean.unsqueeze(1)
    m3 = torch.mean(deviations ** 3, dim=1)
    m2 = std ** 2 + epsilon  # Variance, ensure non-zero with epsilon
    skewness = m3 / (m2 ** 1.5 + epsilon)  # Ensure non-zero denominator

    m4 = torch.mean(deviations ** 4, dim=1)
    kurtosis = m4 / (m2 ** 2 + epsilon) - 3  # Excess kurtosis, ensure non-zero denominator

    # Stack features along a new dimension
    output = torch.stack((mean, var, skewness, kurtosis, std, iqr, max, min, mean, median), dim=1)

    return output


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.sampling_rate = configs.sampling_rate

        self.encoder = statistical_feature_extractor

        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            raise NotImplementedError
        if self.task_name == 'imputation':
            raise NotImplementedError
        if self.task_name == 'anomaly_detection':
            raise NotImplementedError
        if self.task_name == 'classification':
            self.projection = nn.Linear(configs.enc_in*10*6, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raise NotImplementedError

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError

    def anomaly_detection(self, x_enc):
        raise NotImplementedError

    def classification(self, x_enc, x_mark_enc):  # (batch_size, timestamps, enc_in)
        # extract bands
        x_bands_enc = extract_bands(x_enc, self.sampling_rate)
        x_enc = torch.cat((x_enc, x_bands_enc), dim=-1)  # (batch_size, timestamps, enc_in*6)

        enc_out = self.encoder(x_enc)  # (batch_size, features, enc_in*6)
        enc_out = enc_out.reshape(enc_out.shape[0], -1)  # (batch_size, features * enc_in * 6)

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