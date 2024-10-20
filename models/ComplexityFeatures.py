import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from models.StatisticalFeatures import extract_bands


def compute_psd(x_enc):
    fft_result = torch.fft.rfft(x_enc, dim=1)
    psd = torch.abs(fft_result) ** 2
    return psd


def cal_shannon_entropy(psd):
    """
    Calculate Shannon Entropy from PSD.
    """
    epsilon = 1e-9
    norm_psd = psd / (psd.sum(dim=1, keepdim=True) + epsilon)
    log_psd = torch.log2(norm_psd + epsilon)
    entropy = -torch.sum(norm_psd * log_psd, dim=1)
    return entropy


def cal_permutation_entropy(x_enc, m=3, tau=1):
    B, T, C = x_enc.shape
    # Initialize permutation entropy tensor
    pe = torch.zeros(B, C).to(x_enc.device)

    # Due to the complexity of calculating exact permutation patterns,
    # here we provide a simplified approach focusing on conceptual demonstration.
    # For each channel
    for i in range(C):
        # Example: Calculate difference between consecutive points as a proxy to permutation patterns
        diff = x_enc[:, tau:m * tau:tau, i] - x_enc[:, :m * tau - tau:tau, i]
        # Convert differences to binary (1 if positive, 0 if negative or zero)
        binary_diff = (diff > 0).float()
        # Sum binary differences to get a simple measure related to permutation entropy
        pe[:, i] = binary_diff.sum(dim=1)

    # Normalize PE values to be between 0 and 1 (optional, for demonstration)
    pe = pe / (m - 1)

    return pe


def cal_tsallis_entropy(signal, q=2):
    epsilon = 1e-9
    probabilities = signal / (signal.sum(dim=1, keepdim=True) + epsilon)
    tsallis_en = (1 - torch.pow(probabilities, q).sum(dim=1)) / (q - 1)
    return tsallis_en


def cal_spectral_entropy(psd):
    epsilon = 1e-9
    psd_norm = psd / (torch.sum(psd, dim=1, keepdim=True) + epsilon)
    spectral_entropy = -torch.sum(psd_norm * torch.log(psd_norm + epsilon), dim=1)
    return spectral_entropy


def cal_bispectral_entropy(signal):
    B, T, C = signal.shape
    fft_result = torch.fft.fft(signal, dim=1)
    bispectrum = torch.abs(torch.fft.ifft(fft_result * fft_result.conj(), dim=1)) ** 2
    bispectral_entropy = -torch.sum(bispectrum * torch.log(bispectrum + 1e-9), dim=1)
    return bispectral_entropy


def cal_dispersion_entropy(signal, m=2, c=3):
    B, T, C = signal.shape
    def _symbolic_pattern(U):
        pattern = torch.zeros_like(U, dtype=torch.long)
        for i in range(c):
            threshold = torch.quantile(U, (i + 1) / c, dim=1, keepdim=True)
            pattern += (U >= threshold).long()
        return pattern

    pattern = _symbolic_pattern(signal)
    pattern_count = torch.bincount(pattern.view(-1), minlength=c**m)
    probabilities = pattern_count / pattern_count.sum()
    disp_entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9))
    return disp_entropy.repeat(B, C)


def feature_extractor(x_enc, seq_len, m=2, r=None):
    """
    Extract specified entropy features from a frequency band of a batch of trials.
    Map batch with shape BxTxC to BxFxC.
    B is the batch size, T is timestamps, C is channels

    Parameters:
      x_enc : Tensor
        A batch of samples with shape BxTxC
      seq_len : int
        length of the sequence

    Returns:
      output : Tensor
        A batch of entropy features with shape BxFxC
    """

    if r is None:
        r = 0.2 * x_enc.std()

    epsilon = 1e-9  # Small constant to prevent division by zero or log of zero

    # Frequency domain features (PSD)
    psd = compute_psd(x_enc)
    psd_norm = psd / (torch.sum(psd, dim=1, keepdim=True) + epsilon)  # Normalize PSD for probability distribution

    # Spectral Entropy
    spectral_entropy = cal_spectral_entropy(psd)

    # Shannon entropy
    shannon_entropy = cal_shannon_entropy(psd)

    # permutation entropy
    permutation_entropy = cal_permutation_entropy(x_enc)

    # Tsallis entropy
    tsallis_entropy = cal_tsallis_entropy(psd, q=2)

    # Sample entropy
    # sample_entropy = cal_sample_entropy(x_enc, m=m, r=r)

    # Bispectral entropy
    bispectral_entropy = cal_bispectral_entropy(x_enc)

    # Dispersion entropy
    dispersion_entropy = cal_dispersion_entropy(x_enc, m=m, c=3)

    output = torch.stack((spectral_entropy, tsallis_entropy, shannon_entropy, permutation_entropy,
                          bispectral_entropy, dispersion_entropy), dim=1)

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
            self.projection = nn.Linear(configs.enc_in*6*6, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raise NotImplementedError

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError

    def anomaly_detection(self, x_enc):
        raise NotImplementedError

    def classification(self, x_enc, x_mark_enc):  # (batch_size, timestamps, enc_in)
        # extract bands
        x_bands_enc = extract_bands(x_enc, self.sampling_rate)
        x_enc = torch.cat((x_enc, x_bands_enc), dim=-1)  # (batch_size, timestamps, enc_in * 6)

        enc_out = self.encoder(x_enc, self.sampling_rate)  # (batch_size, features, enc_in * 6)
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