import torch

from ams.objective.masking import inverse_block_masking, num_visible_tokens


def test_inverse_block_masking_is_reproducible_and_fixed_count() -> None:
    kwargs = dict(
        batch_size=3,
        time_patches=8,
        frequency_patches=8,
        mask_ratio=0.8,
        block_size=(2, 2),
        device=torch.device("cpu"),
    )
    first = inverse_block_masking(**kwargs, generator=torch.Generator().manual_seed(9))
    second = inverse_block_masking(**kwargs, generator=torch.Generator().manual_seed(9))
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)
    visible, removal, _, kept = first
    assert torch.all(
        visible.sum(1) == num_visible_tokens(time_patches=8, frequency_patches=8, mask_ratio=0.8)
    )
    assert torch.equal(visible, torch.zeros_like(visible).scatter(1, kept, True))
    assert torch.equal(removal, ~visible)
