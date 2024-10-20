import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def compute_psd(x_enc):
    fft_result = torch.fft.rfft(x_enc, dim=1)
    psd = torch.abs(fft_result) ** 2
    return psd


def compute_band_power(psd, fs, band):
    freqs = torch.linspace(0, fs / 2, steps=psd.shape[1])  # Adjusted to match PSD shape
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    # print(band_mask)
    band_power = torch.sum(psd[:, band_mask, :], dim=1)
    return band_power


def feature_extractor(x_enc, fs):
    """
    Extract features from a frequency band of a batch of trials. Map batch with shape BxTxC to BxFxC.
    B is the batch size, T is timestamps, C is channels
    Calculate Power Spectral Density and some PSD related biomarkers

    Parameters:
      x_enc : Tensor
        A batch of samples with shape BxTxC
      fs : int
        Sampling frequency rate

    Returns:
      output : Tensor
        A batch of psd with shape Bx(T/2+12)xC
    """
    epsilon = 1e-9  # Small constant to prevent division by zero or log of zero

    # Frequency domain features (PSD)
    psd = compute_psd(x_enc)

    # Compute band powers for α and β bands
    delta_power = compute_band_power(psd, fs, (0.5, 4))
    theta_power = compute_band_power(psd, fs, (4, 8))
    alpha_power = compute_band_power(psd, fs, (8, 12))
    beta_power = compute_band_power(psd, fs, (12, 30))

    # Compute total power for normalization (to compute relative power)
    total_power = torch.sum(psd, dim=1) + epsilon

    # Compute relative power for bands
    delta_rel_power = delta_power / total_power
    theta_rel_power = theta_power / total_power
    alpha_rel_power = alpha_power / total_power
    beta_rel_power = beta_power / total_power

    # Compute ratios of EEG rhythms
    theta_alpha_ratio = theta_power / (alpha_power + epsilon)
    alpha_beta_ratio = alpha_power / (beta_power + epsilon)

    features = torch.stack((delta_power, theta_power, alpha_power, beta_power, total_power,
                            theta_alpha_ratio, alpha_beta_ratio,
                            delta_rel_power, theta_rel_power, alpha_rel_power, beta_rel_power), dim=1)

    # Stack features with raw psd
    output = torch.cat((psd, features), dim=1)
    # output = features

    return output


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.sampling_rate = configs.sampling_rate

        self.encoder = feature_extractor

        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            raise NotImplementedError
        if self.task_name == 'imputation':
            raise NotImplementedError
        if self.task_name == 'anomaly_detection':
            raise NotImplementedError
        if self.task_name == 'classification':
            self.projection = nn.Linear(configs.enc_in * (math.floor(self.seq_len / 2) + 12), configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raise NotImplementedError

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError

    def anomaly_detection(self, x_enc):
        raise NotImplementedError

    def classification(self, x_enc, x_mark_enc):  # (batch_size, timestamps, enc_in)
        # extract bands
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