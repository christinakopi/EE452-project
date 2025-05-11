import numpy as np
from scipy import signal
import pywt

bp_filter = signal.butter(4, (0.5, 30), btype="bandpass", output="sos", fs=250)

# filtering for the signals (given at example.ipynb)
def time_filtering(x: np.ndarray) -> np.ndarray:
    """Filter signal in the time domain"""
    return signal.sosfiltfilt(bp_filter, x, axis=0).copy()


def fft_filtering(x: np.ndarray) -> np.ndarray:
    """Compute FFT and only keep"""
    x = np.abs(np.fft.fft(x, axis=0))
    x = np.log(np.where(x > 1e-8, x, 1e-8))

    win_len = x.shape[0]
    # Only frequencies b/w 0.5 and 30Hz
    return x[int(0.5 * win_len // 250) : 30 * win_len // 250]


def stft_filtering(x: np.ndarray) -> np.ndarray:
    nperseg = x.shape[0]
    channel_shapes = x.shape[1]
    f, t, Zxx = signal.stft(x, fs=250, axis=0, nperseg=nperseg, noverlap=0)

    Zxx = np.abs(Zxx.reshape(-1, channel_shapes))
    mag = np.log(np.where(Zxx > 1e-8, Zxx, 1e-8))

    win_len = mag.shape[0]
    # Only frequencies b/w 0.5 and 30Hz
    return mag[int(0.5 * win_len // 250) : 30 * win_len // 250]


def psd_filtering(x: np.ndarray) -> np.ndarray:
    nperseg = x.shape[0]
    f, Pxx = signal.welch(x, fs=250, axis=0, nperseg=nperseg, noverlap=0)
    Pxx = np.abs(Pxx)
    mag = np.log(np.where(Pxx > 1e-8, Pxx, 1e-8))

    win_len = mag.shape[0]
    # Only frequencies b/w 0.5 and 30Hz
    return mag[int(0.5 * win_len // 250) : 30 * win_len // 250]


def wt_filtering(x: np.ndarray,  wavelet='db4', threshold=0)-> np.ndarray :
    denoised = np.zeros_like(x)
    orig_len, n_signals = x.shape

    for i in range(n_signals):
        coeffs = pywt.wavedec(x[:, i], wavelet, level=1)

        for j in range(1, len(coeffs)):
            K = np.round(threshold * len(coeffs[j])).astype(int)
            if K < len(coeffs[j]):
                coeffs[j][K:] = 0

        denoised_ = pywt.waverec(coeffs, wavelet)

        # handle length mismatch
        if len(denoised_) > orig_len:
            denoised[:, i] = denoised_[:orig_len]
        elif len(denoised_) < orig_len:
            denoised[:, i] = np.pad(denoised_, (0, orig_len - len(denoised_)), 'constant')
        else:
            denoised[:, i] = denoised_

    return denoised