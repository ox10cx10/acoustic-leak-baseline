"""93 acoustic descriptors for one-second recordings sampled at 8192 Hz."""

from __future__ import annotations

from numbers import Real

import numpy as np
from scipy.signal import stft
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

SAMPLE_RATE = 8192
N_SAMPLES = 8192
FEATURE_VERSION = "shape93-v1"
_TINY = np.finfo(np.float64).tiny
_BANDS = ((0, 100), (100, 300), (300, 600), (600, 1000), (1000, 2000), (2000, np.inf))
_DYNAMIC_BANDS = ((0, 300), (300, 1000), (1000, 2000), (2000, 4097))
_GLOBAL_NAMES = (
    "log10_crest_factor", "crest_factor", "zero_crossing_rate",
    "spectral_centroid_hz", "spectral_bandwidth_hz", "spectral_flatness",
    "spectral_entropy", "spectral_rolloff_85_hz", "dominant_frequency_hz",
) + tuple(
    f"band_fraction_{low}_{'nyquist' if np.isinf(high) else high}_hz"
    for low, high in _BANDS
)
_LOGMEL_NAMES = tuple(
    f"logmel_{stat}_{band:02d}" for stat in ("mean", "std") for band in range(32)
)
_DYNAMIC_NAMES = (
    "frame_log_energy_std", "frame_log_energy_p90_p10", "frame_energy_modulation_peak",
    "spectral_flux_mean", "spectral_flux_std", "frame_centroid_std_hz",
    "frame_flatness_mean", "frame_flatness_std",
    "top5_bin_concentration_mean", "top5_bin_concentration_std",
) + tuple(
    f"frame_band_fraction_std_{low}_{min(high, SAMPLE_RATE // 2)}_hz"
    for low, high in _DYNAMIC_BANDS
)
FEATURE_NAMES = _GLOBAL_NAMES + _LOGMEL_NAMES + _DYNAMIC_NAMES


def _validate(signals: np.ndarray, sample_rate: float) -> np.ndarray:
    if (
        isinstance(sample_rate, (bool, np.bool_))
        or not isinstance(sample_rate, Real)
        or sample_rate != SAMPLE_RATE
    ):
        raise ValueError("Input must be sampled at 8192 Hz; resampling is not automatic")
    signals = np.asarray(signals)
    if signals.ndim != 2 or signals.shape[0] == 0 or signals.shape[1] != N_SAMPLES:
        raise ValueError("Input must have shape (n_recordings, 8192), with at least one recording")
    if not np.issubdtype(signals.dtype, np.number) or np.iscomplexobj(signals):
        raise ValueError("Recordings must contain real numeric samples")
    signals = np.asarray(signals, dtype=np.float64)
    if not np.isfinite(signals).all():
        raise ValueError("Recordings must contain finite samples")
    if np.any(np.all(signals == signals[:, :1], axis=1)):
        raise ValueError("Recordings must contain nonconstant AC sound")
    return signals


def _normalize(signals: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        centered = signals - signals.mean(axis=1, keepdims=True)
    peak = np.max(np.abs(centered), axis=1, keepdims=True)
    if not np.isfinite(centered).all() or np.any(peak <= 0.0):
        raise ValueError("Centering must produce finite, nonconstant AC sound")
    # Scaling before squaring avoids underflow for quiet, nonzero recordings.
    scaled = centered / peak
    return scaled / np.sqrt(np.mean(scaled**2, axis=1, keepdims=True))


def _global_features(signals: np.ndarray) -> np.ndarray:
    centered = signals - signals.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=1))
    peak = np.max(np.abs(centered), axis=1)
    crest = peak / np.maximum(rms, _TINY)
    zcr = np.mean(centered[:, :-1] * centered[:, 1:] < 0.0, axis=1)
    frequencies = np.fft.rfftfreq(N_SAMPLES, d=1.0 / SAMPLE_RATE)
    power = np.abs(np.fft.rfft(centered * np.hanning(N_SAMPLES), axis=1)) ** 2
    total = np.maximum(power.sum(axis=1), _TINY)
    probability = power / total[:, None]
    centroid = (power @ frequencies) / total
    bandwidth = np.sqrt(
        np.sum(power * (frequencies[None, :] - centroid[:, None]) ** 2, axis=1) / total
    )
    flatness = np.exp(np.mean(np.log(np.maximum(power, _TINY)), axis=1))
    flatness /= np.maximum(np.mean(power, axis=1), _TINY)
    entropy = -np.sum(probability * np.log(np.maximum(probability, _TINY)), axis=1)
    entropy /= np.log(power.shape[1])
    rolloff = frequencies[np.argmax(np.cumsum(power, axis=1) >= 0.85 * total[:, None], axis=1)]
    dominant = frequencies[np.argmax(power[:, 1:], axis=1) + 1]
    columns = [
        np.log10(np.maximum(peak, _TINY)), crest, zcr, centroid, bandwidth,
        flatness, entropy, rolloff, dominant,
    ]
    for low, high in _BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        columns.append(power[:, mask].sum(axis=1) / total)
    return np.column_stack(columns)


def _frame_power(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = signals - signals.mean(axis=1, keepdims=True)
    frequencies, _, spectrum = stft(
        centered, fs=SAMPLE_RATE, window="hann", nperseg=256, noverlap=128,
        nfft=256, detrend=False, return_onesided=True, boundary=None,
        padded=False, axis=1,
    )
    return frequencies, np.abs(spectrum) ** 2


def _logmel_features(power: np.ndarray) -> np.ndarray:
    frequencies = np.fft.rfftfreq(256, d=1.0 / SAMPLE_RATE)
    mel_min = 2595.0 * np.log10(1.0 + np.asarray(20.0) / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + np.asarray(SAMPLE_RATE / 2.0) / 700.0)
    mel_points = np.linspace(mel_min, mel_max, 34)
    edges = 700.0 * (10.0 ** (np.asarray(mel_points) / 2595.0) - 1.0)
    filters = np.zeros((32, frequencies.size), dtype=np.float64)
    for band in range(32):
        left, center, right = edges[band:band + 3]
        rising = (frequencies - left) / max(center - left, np.finfo(float).eps)
        falling = (right - frequencies) / max(right - center, np.finfo(float).eps)
        filters[band] = np.maximum(0.0, np.minimum(rising, falling))
        filters[band] /= filters[band].sum()
    mel_power = np.einsum("mf,nft->nmt", filters, power, optimize=True)
    floor = np.maximum(np.max(mel_power, axis=(1, 2), keepdims=True) * 1e-12, _TINY)
    logmel = np.log(mel_power + floor)
    return np.column_stack([logmel.mean(axis=2), logmel.std(axis=2)])


def _dynamic_features(power: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    energy = power.sum(axis=1)
    floor = np.maximum(np.max(energy, axis=1, keepdims=True) * 1e-12, _TINY)
    log_energy = np.log(energy + floor)
    centered_energy = log_energy - log_energy.mean(axis=1, keepdims=True)
    correlations = []
    for lag in range(1, min(20, log_energy.shape[1] - 1) + 1):
        left, right = centered_energy[:, :-lag], centered_energy[:, lag:]
        numerator = np.sum(left * right, axis=1)
        denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
        correlations.append(numerator / np.maximum(denominator, 1e-15))
    normalized_power = power / np.maximum(energy[:, None, :], _TINY)
    flux = np.sum(np.diff(normalized_power, axis=2) ** 2, axis=1)
    centroid = np.sum(normalized_power * frequencies[None, :, None], axis=1)
    bin_floor = np.maximum(np.max(power, axis=1, keepdims=True) * 1e-12, _TINY)
    flatness = np.exp(np.mean(np.log(power + bin_floor), axis=1))
    flatness /= np.maximum(np.mean(power + bin_floor, axis=1), _TINY)
    concentration = np.partition(normalized_power, -5, axis=1)[:, -5:, :].sum(axis=1)
    columns = [
        log_energy.std(axis=1),
        np.quantile(log_energy, 0.9, axis=1) - np.quantile(log_energy, 0.1, axis=1),
        np.max(np.column_stack(correlations), axis=1),
        flux.mean(axis=1), flux.std(axis=1), centroid.std(axis=1),
        flatness.mean(axis=1), flatness.std(axis=1),
        concentration.mean(axis=1), concentration.std(axis=1),
    ]
    for low, high in _DYNAMIC_BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        columns.append(normalized_power[:, mask, :].sum(axis=1).std(axis=1))
    return np.column_stack(columns)


def extract_features(
    signals: np.ndarray, sample_rate: float = SAMPLE_RATE,
) -> tuple[np.ndarray, list[str]]:
    """Extract 15 global, 64 log-Mel and 14 temporal descriptors per recording.

    Input is a nonempty matrix of one-second recordings at 8192 Hz. Each row
    is centered and RMS-normalized independently. The output has shape
    (n_recordings, 93); no parameters are learned from other recordings.
    """
    normalized = _normalize(_validate(signals, sample_rate))
    frequencies, power = _frame_power(normalized)
    features = np.column_stack([
        _global_features(normalized), _logmel_features(power),
        _dynamic_features(power, frequencies),
    ])
    if not np.isfinite(features).all():
        raise ValueError("Feature extraction produced non-finite values")
    return features, list(FEATURE_NAMES)


class AcousticFeatures(TransformerMixin, BaseEstimator):
    """Scikit-learn transformer for fixed-format acoustic recordings.

    Fit validates the waveform format; feature extraction has no learned
    statistics. Put a StandardScaler after this transformer in a Pipeline so
    that scaling parameters are fitted only on the training partition.
    """

    def __init__(self, sample_rate: float = SAMPLE_RATE):
        self.sample_rate = sample_rate

    def fit(self, X: np.ndarray, y=None):
        signals = _validate(X, self.sample_rate)
        _normalize(signals)
        self.n_features_in_ = signals.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        signals = _validate(X, self.sample_rate)
        if signals.shape[1] != self.n_features_in_:
            raise ValueError("Recording length must match the fitted waveform format")
        return extract_features(signals, self.sample_rate)[0]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        if input_features is not None and np.asarray(input_features).shape != (self.n_features_in_,):
            raise ValueError("input_features must name every waveform sample")
        return np.asarray(FEATURE_NAMES, dtype=object)
