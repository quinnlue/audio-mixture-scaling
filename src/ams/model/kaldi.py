"""Pure-PyTorch Kaldi fbank matching torchaudio.compliance.kaldi defaults.

Vendored from torchaudio (BSD) without the torchaudio native extension dependency.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_EPSILON = torch.tensor(torch.finfo(torch.float).eps)
_MILLISECONDS_TO_SECONDS = 0.001
_HANNING = "hanning"


def _epsilon(device: torch.device, dtype: torch.dtype) -> Tensor:
    return _EPSILON.to(device=device, dtype=dtype)


def _next_power_of_2(value: int) -> int:
    return 1 if value == 0 else 2 ** (value - 1).bit_length()


def _get_strided(waveform: Tensor, window_size: int, window_shift: int, snip_edges: bool) -> Tensor:
    assert waveform.dim() == 1
    num_samples = waveform.size(0)
    strides = (window_shift * waveform.stride(0), waveform.stride(0))

    if snip_edges:
        if num_samples < window_size:
            return torch.empty((0, 0), dtype=waveform.dtype, device=waveform.device)
        m = 1 + (num_samples - window_size) // window_shift
    else:
        reversed_waveform = torch.flip(waveform, [0])
        m = (num_samples + (window_shift // 2)) // window_shift
        pad = window_size // 2 - window_shift // 2
        pad_right = reversed_waveform
        if pad > 0:
            pad_left = reversed_waveform[-pad:]
            waveform = torch.cat((pad_left, waveform, pad_right), dim=0)
        else:
            waveform = torch.cat((waveform[-pad:], pad_right), dim=0)

    return waveform.as_strided((m, window_size), strides)


def _feature_window_function(
    window_type: str,
    window_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if window_type == _HANNING:
        return torch.hann_window(window_size, periodic=False, device=device, dtype=dtype)
    raise ValueError(f"Unsupported window_type {window_type!r}")


def _get_log_energy(strided_input: Tensor, epsilon: Tensor, energy_floor: float) -> Tensor:
    log_energy = torch.max(strided_input.pow(2).sum(1), epsilon).log()
    if energy_floor == 0.0:
        return log_energy
    return torch.max(
        log_energy,
        torch.tensor(
            math.log(energy_floor), device=strided_input.device, dtype=strided_input.dtype
        ),
    )


def _get_waveform_and_window_properties(
    waveform: Tensor,
    channel: int,
    sample_frequency: float,
    frame_shift: float,
    frame_length: float,
    round_to_power_of_two: bool,
    preemphasis_coefficient: float,
) -> tuple[Tensor, int, int, int]:
    channel = max(channel, 0)
    waveform = waveform[channel, :]
    window_shift = int(sample_frequency * frame_shift * _MILLISECONDS_TO_SECONDS)
    window_size = int(sample_frequency * frame_length * _MILLISECONDS_TO_SECONDS)
    padded_window_size = _next_power_of_2(window_size) if round_to_power_of_two else window_size
    return waveform, window_shift, window_size, padded_window_size


def _get_window(
    waveform: Tensor,
    padded_window_size: int,
    window_size: int,
    window_shift: int,
    window_type: str,
    snip_edges: bool,
    raw_energy: bool,
    energy_floor: float,
    dither: float,
    remove_dc_offset: bool,
    preemphasis_coefficient: float,
) -> tuple[Tensor, Tensor]:
    device, dtype = waveform.device, waveform.dtype
    epsilon = _epsilon(device, dtype)
    strided_input = _get_strided(waveform, window_size, window_shift, snip_edges)

    if dither != 0.0:
        strided_input = (
            strided_input + torch.randn(strided_input.shape, device=device, dtype=dtype) * dither
        )

    if remove_dc_offset:
        strided_input = strided_input - torch.mean(strided_input, dim=1).unsqueeze(1)

    if raw_energy:
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)
    else:
        signal_log_energy = torch.empty(0, device=device, dtype=dtype)

    if preemphasis_coefficient != 0.0:
        offset = torch.nn.functional.pad(
            strided_input.unsqueeze(0), (1, 0), mode="replicate"
        ).squeeze(0)
        strided_input = strided_input - preemphasis_coefficient * offset[:, :-1]

    window_function = _feature_window_function(window_type, window_size, device, dtype).unsqueeze(0)
    strided_input = strided_input * window_function

    if padded_window_size != window_size:
        strided_input = torch.nn.functional.pad(
            strided_input.unsqueeze(0),
            (0, padded_window_size - window_size),
            mode="constant",
            value=0,
        ).squeeze(0)

    if not raw_energy:
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)

    return strided_input, signal_log_energy


def inverse_mel_scale(mel_freq: Tensor) -> Tensor:
    return 700.0 * ((mel_freq / 1127.0).exp() - 1.0)


def mel_scale(freq: Tensor) -> Tensor:
    return 1127.0 * (1.0 + freq / 700.0).log()


def mel_scale_scalar(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)


def get_mel_banks(
    num_bins: int,
    window_length_padded: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
) -> tuple[Tensor, Tensor]:
    num_fft_bins = window_length_padded / 2
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq += nyquist

    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = mel_scale_scalar(low_freq)
    mel_high_freq = mel_scale_scalar(high_freq)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    bin_idx = torch.arange(num_bins).unsqueeze(1)
    left_mel = mel_low_freq + bin_idx * mel_freq_delta
    center_mel = mel_low_freq + (bin_idx + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (bin_idx + 2.0) * mel_freq_delta
    center_freqs = inverse_mel_scale(center_mel)
    mel = mel_scale(fft_bin_width * torch.arange(num_fft_bins)).unsqueeze(0)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    bins = torch.max(torch.zeros(1), torch.min(up_slope, down_slope))
    return bins, center_freqs


def fbank(
    waveform: Tensor,
    *,
    sample_frequency: float = 16000.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    num_mel_bins: int = 128,
    dither: float = 0.0,
    htk_compat: bool = True,
    use_energy: bool = False,
    window_type: str = _HANNING,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
) -> Tensor:
    """Compute Kaldi filter-bank features for waveform shape ``(C, T)``."""
    device, dtype = waveform.device, waveform.dtype
    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        waveform,
        channel=-1,
        sample_frequency=sample_frequency,
        frame_shift=frame_shift,
        frame_length=frame_length,
        round_to_power_of_two=True,
        preemphasis_coefficient=0.97,
    )
    if len(waveform) == 0:
        return torch.empty(0, device=device, dtype=dtype)

    strided_input, signal_log_energy = _get_window(
        waveform,
        padded_window_size,
        window_size,
        window_shift,
        window_type,
        snip_edges=True,
        raw_energy=True,
        energy_floor=1.0,
        dither=dither,
        remove_dc_offset=True,
        preemphasis_coefficient=0.97,
    )

    spectrum = torch.fft.rfft(strided_input).abs().pow(2.0)
    mel_energies, _ = get_mel_banks(
        num_mel_bins,
        padded_window_size,
        sample_frequency,
        low_freq,
        high_freq,
    )
    mel_energies = mel_energies.to(device=device, dtype=dtype)
    mel_energies = torch.nn.functional.pad(mel_energies, (0, 1), mode="constant", value=0)
    mel_energies = torch.mm(spectrum, mel_energies.T)
    mel_energies = torch.max(mel_energies, _epsilon(device, dtype)).log()

    if use_energy:
        signal_log_energy = signal_log_energy.unsqueeze(1)
        if htk_compat:
            mel_energies = torch.cat((mel_energies, signal_log_energy), dim=1)
        else:
            mel_energies = torch.cat((signal_log_energy, mel_energies), dim=1)
    return mel_energies


def fbank_batch(
    waveform: Tensor,
    *,
    sample_frequency: float = 16000.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    num_mel_bins: int = 128,
    dither: float = 0.0,
    htk_compat: bool = True,
    use_energy: bool = False,
    window_type: str = _HANNING,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
) -> Tensor:
    """Batched Kaldi fbank for waveform shape ``(B, T)`` -> ``(B, frames, mels)``.

    Numerically identical to running :func:`fbank` on each row of ``(1, T)``
    (with ``dither=0``): same snip-edges framing, DC removal, raw-energy floor,
    0.97 pre-emphasis, Hann window, power-of-two FFT and mel projection—just
    vectorized over the batch with ``unfold`` + batched ``rfft``/``matmul`` so
    the per-sample Python loop and its device syncs disappear.
    """
    if waveform.dim() != 2:
        raise ValueError(f"fbank_batch expects (B, T), got shape {tuple(waveform.shape)}")
    device, dtype = waveform.device, waveform.dtype
    window_shift = int(sample_frequency * frame_shift * _MILLISECONDS_TO_SECONDS)
    window_size = int(sample_frequency * frame_length * _MILLISECONDS_TO_SECONDS)
    padded_window_size = _next_power_of_2(window_size)

    batch, num_samples = waveform.shape
    if num_samples < window_size:
        return torch.empty((batch, 0, num_mel_bins), device=device, dtype=dtype)

    epsilon = _epsilon(device, dtype)
    # snip_edges framing: (B, frames, window_size)
    strided = waveform.unfold(1, window_size, window_shift)

    if dither != 0.0:
        strided = strided + torch.randn(strided.shape, device=device, dtype=dtype) * dither

    # remove_dc_offset
    strided = strided - strided.mean(dim=2, keepdim=True)

    # raw energy (before pre-emphasis/window), energy_floor=1.0 -> floor at log(1)=0
    if use_energy:
        log_energy = torch.max(strided.pow(2).sum(2), epsilon).log()
        log_energy = torch.max(log_energy, torch.zeros((), device=device, dtype=dtype))

    # pre-emphasis 0.97 with replicate padding on the frame axis
    offset = torch.nn.functional.pad(strided, (1, 0), mode="replicate")
    strided = strided - 0.97 * offset[..., :-1]

    window = _feature_window_function(window_type, window_size, device, dtype)
    strided = strided * window

    if padded_window_size != window_size:
        strided = torch.nn.functional.pad(
            strided, (0, padded_window_size - window_size), mode="constant", value=0
        )

    spectrum = torch.fft.rfft(strided).abs().pow(2.0)  # (B, frames, padded//2 + 1)
    mel_banks, _ = get_mel_banks(
        num_mel_bins, padded_window_size, sample_frequency, low_freq, high_freq
    )
    mel_banks = mel_banks.to(device=device, dtype=dtype)
    mel_banks = torch.nn.functional.pad(mel_banks, (0, 1), mode="constant", value=0)
    mel_energies = torch.matmul(spectrum, mel_banks.T)
    mel_energies = torch.max(mel_energies, epsilon).log()

    if use_energy:
        log_energy = log_energy.unsqueeze(2)
        if htk_compat:
            mel_energies = torch.cat((mel_energies, log_energy), dim=2)
        else:
            mel_energies = torch.cat((log_energy, mel_energies), dim=2)
    return mel_energies


__all__ = ["fbank", "fbank_batch"]
