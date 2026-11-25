"""Fingerprint interface.

A record is compressed into a fixed-length vector so that records with
different channel counts can be compared in one space.  The paper's
fingerprint is ``five frozen-encoder scalars + three band ratios`` (eight
dimensions; the ERP path adds a drift ratio and a frontal ratio, ten
dimensions).  The encoder is an external frozen model supplied by the reader;
this module only provides the band-ratio example and a concatenation helper.
No encoder weights, thresholds, or paper values are included.
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np


class Fingerprinter(Protocol):
    """``record -> fixed-length vector``."""

    def __call__(self, record: np.ndarray, fs: float) -> np.ndarray:  # pragma: no cover
        ...


def _band_power(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """Mean power in [lo, hi] Hz using a Hann-windowed periodogram."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 8:
        return 0.0
    x = x - x.mean()
    w = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * w)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    band = (freqs >= lo) & (freqs < hi)
    return float(spec[band].mean()) if band.any() else 0.0


def band_ratio_fingerprint(
    record: np.ndarray,
    fs: float,
    line_hz: float,
    line_band: float,
    emg_band: Sequence[float],
    ref_band: Sequence[float],
) -> np.ndarray:
    """Generic three-band-ratio fingerprint.

    Returns ``[line_noise_ratio, emg_ratio, channel_outlier_ratio]`` computed
    as ratios of mean band power.  Every band is a required argument: the
    framework carries no default band configuration.  This is an *example*
    implementation, not the paper's value.

    Parameters and literature recommendations
    -----------------------------------------
    line_hz : float
        Power-line frequency.  Literature uses 50 or 60 Hz depending on the
        local grid (see the notch and line-noise entries in REFERENCES.md).
    line_band : float
        Half-width around the line frequency.  A narrow notch is the
        conventional choice (Ferdjallah & Barr, 1994).
    emg_band : (float, float)
        40-100 Hz is commonly used for EMG band power
        (Fitzgibbon et al., 2016; Pope et al., 2022).
    ref_band : (float, float)
        8-30 Hz covers mu (8-13 Hz) and beta (13-30 Hz) for motor imagery.

    ``channel_outlier_ratio`` is the maximum channel RMS divided by the median
    channel RMS when the record is multi-channel, and 1.0 otherwise.
    """
    x = np.asarray(record, dtype=float)
    if x.ndim == 1:
        x = x[None, :]

    line = _band_power(x, fs, line_hz - line_band, line_hz + line_band)
    flanks = 0.5 * (
        _band_power(x, fs, line_hz - 3 * line_band, line_hz - line_band)
        + _band_power(x, fs, line_hz + line_band, line_hz + 3 * line_band)
    )
    line_ratio = line / flanks if flanks > 0 else 0.0

    emg = _band_power(x, fs, emg_band[0], emg_band[1])
    ref = _band_power(x, fs, ref_band[0], ref_band[1])
    emg_ratio = emg / ref if ref > 0 else 0.0

    rms = np.sqrt((x ** 2).mean(axis=1))
    median_rms = float(np.median(rms))
    channel_ratio = float(rms.max() / median_rms) if median_rms > 0 else 1.0

    return np.array([line_ratio, emg_ratio, channel_ratio], dtype=float)


class CallableFingerprinter:
    """Wrap any ``record, fs -> vector`` callable."""

    def __init__(self, fn: Callable[[np.ndarray, float], np.ndarray]):
        self.fn = fn

    def __call__(self, record: np.ndarray, fs: float) -> np.ndarray:
        return np.asarray(self.fn(record, fs), dtype=float)


def concat_fingerprint(*parts: np.ndarray) -> np.ndarray:
    """Concatenate encoder scalars and band ratios into one vector.

    The caller decides the order and length; e.g. five encoder scalars plus
    three band ratios give the eight-dimensional fingerprint, and the ERP
    path appends a drift ratio and a frontal ratio.
    """
    return np.concatenate([np.asarray(p, dtype=float).ravel() for p in parts])
