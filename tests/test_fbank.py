import torch

from ams.model.kaldi import fbank, fbank_batch


def test_batched_fbank_matches_single_row_reference() -> None:
    waveform = torch.randn(3, 1600)
    batched = fbank_batch(waveform, dither=0.0)
    expected = torch.stack([fbank(row.unsqueeze(0), dither=0.0) for row in waveform])
    torch.testing.assert_close(batched, expected, rtol=1e-5, atol=1e-5)
