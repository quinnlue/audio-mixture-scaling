# Code Style

This guide covers aesthetic code style and inline documentation for `src/audiossl`.
It does not define runtime behavior, validation policy, exception policy, or
component contracts. Those belong in `docs/contracts`.

## Principles

Prefer code that reads plainly before adding comments. Names, type annotations,
dataclasses, and small helper functions should carry most of the meaning.

Use documentation where it preserves information that the code cannot easily
show:

- Public API purpose and usage.
- Config intent, units, and important field relationships.
- Tensor shapes, mask semantics, and layout conventions.
- Algorithm choices, external parity, or paper/model compatibility.
- Lifecycle details that are easy to break during maintenance.

Avoid comments that restate the next line of code.

## Public API Docstrings

Every public module, class, function, and public method in `src/audiossl` should
have a docstring when it is part of the user-facing or cross-component API.

Treat a symbol as public when it is exported, imported by another package area,
documented in `docs/contracts`, exposed through presets/pretrained APIs, or used
as a stage boundary such as `AudioModel.process`, `AudioModel.tokenize`, or
`Objective.metrics`.

Use NumPy-style docstrings for public APIs with meaningful parameters, return
values, or notes.

```python
class Trainer:
    """Single-device SSL trainer.

    Parameters
    ----------
    model
        Pre-built audio model. Batches are moved to the model runtime device.
    objective
        Training objective that owns the SSL-specific forward pass and loss.
    train_loader
        Loader that yields `AudioBatch` instances.

    Notes
    -----
    The trainer is intentionally single-device. Distributed wrapping is outside
    this class.
    """
```

Compact docstrings are fine when the API is simple.

```python
def masked_mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool valid time steps using a boolean validity mask."""
```

## Config Docstrings

All config dataclasses should have docstrings. The goal is not to repeat every
field name. Explain what the config is for, the expected units, and any field
relationships that affect how the component is read.

```python
@dataclass(kw_only=True, frozen=True)
class ConvModuleConfig:
    """Conformer-style depthwise-separable convolution module.

    `kernel_size` is measured over the time axis. `expansion_factor` controls
    the pointwise expansion before GLU, so even values preserve a clean channel
    split.
    """

    kernel_size: int
    expansion_factor: int = 2
```

One-line docstrings are enough for obvious grouping configs.

```python
@dataclass(kw_only=True, frozen=True)
class BranchMergeConfig:
    """Merge policy for the parallel attention and cgMLP branches."""
```

## Private Helper Docstrings

Private helpers do not need docstrings by default. Add one only when the helper
is load-bearing:

- It encodes a tensor layout or mask convention.
- It mirrors an external implementation.
- It implements a non-obvious algorithm.
- It protects a public API from duplicated complexity.
- It is likely to be edited incorrectly without local context.

```python
def _convert_patch_embed_weight(conv_weight: torch.Tensor) -> torch.Tensor:
    """Convert Dasheng patch weights into this repo's time-major token order."""
```

## Tensor Shape Notation

Use parenthesized shape notation in docstrings and comments:

- `B`: batch size.
- `S`: raw waveform samples.
- `T`: time frames or time tokens.
- `F`: feature or frequency bins.
- `N`: flattened token count.
- `D`: embedding/model dimension.
- `C`: class count or channel count, whichever is local and obvious.
- `G`: groups, codebooks, or grouped-token count.
- `K`: kernel size, neighbors, or sampled candidates.

Prefer explicit names when a letter would be ambiguous:

- `(B, S)` for raw waveform batches.
- `(B, T, F)` for feature sequences.
- `(B, N, D)` for token embeddings.
- `(B, T)` or `(B, N)` for boolean validity masks.
- `(num_masked, 1 + K)` for contrastive logits.

State mask semantics whenever they are not the default. The default convention
is: boolean masks use `True` for valid positions.

```python
def random_masking(
    x: torch.Tensor,
    mask_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token random masking.

    Parameters
    ----------
    x
        Token embeddings with shape `(B, N, D)`.
    mask_ratio
        Fraction of tokens to remove.

    Returns
    -------
    x_masked
        Kept tokens with shape `(B, len_keep, D)`.
    mask
        Float mask with shape `(B, N)`, where `1.0` means removed.
    ids_restore
        Inverse permutation with shape `(B, N)`.
    """
```

For inline comments, keep shape comments close to the transformation they
explain.

```python
# Unfold to `(B, T_patches, F_patches, patch_t, patch_f)`, then flatten in
# time-major order to `(B, N, patch_t * patch_f)`.
x = x.unfold(1, patch_t, stride_t).unfold(2, patch_f, stride_f).contiguous()
x = x.reshape(batch_size, num_time_patches * num_feat_patches, patch_t * patch_f)
```

## Inline Comments

Use inline comments sparingly. Good comments answer "why this shape, why this
order, why this timing, why this compatibility choice?"

Good:

```python
# Match Dasheng: `len_keep` is not rounded to a group boundary, so the
# kept/masked split may divide a group.
len_keep = int(num_tokens * (1.0 - mask_ratio))
```

Not useful:

```python
# Calculate len_keep.
len_keep = int(num_tokens * (1.0 - mask_ratio))
```

Comments should be complete sentences with normal capitalization. Prefer one or
two short lines over a paragraph. If a comment grows large, consider a helper
function or a docstring.

## Module Docstrings

Public modules should have a short module docstring when the filename alone does
not explain the role of the module.

Good candidates:

- `training/trainer.py`
- `pretrained/load.py`
- `integrations/*/porter.py`
- Objective, processor, tokenizer, and evaluation modules with public APIs.

Tiny dispatch modules, compatibility shims, and `__init__.py` files can use
one-line docstrings or no docstring when the package exports are obvious.

## Examples

The trainer module is a good model for public class documentation: a concise
module docstring plus a NumPy-style class docstring.

`src/audiossl/training/trainer.py`

The MAE masking helpers are a good model for documenting external parity and
return-shape semantics.

`src/audiossl/modelling/objectives/mae/masking.py`

The patch tokenizer docstring is the right kind of documentation for a
shape-heavy component. Keep this style, but prefer ASCII punctuation and the
standard shape notation in new code.

`src/audiossl/modelling/tokenizers/patchify.py`

## Avoid

Avoid large banner comments. Prefer short section comments only when they make a
long file easier to scan, and keep them plain.

```python
# Public methods
```

Avoid visual dividers.

```python
# -----------------------------------------------------Runtime Config-----------------------------------------------------
```

Avoid dead commented-out code. Delete it, or leave a short comment explaining
the active design choice.

```python
# Bad: commented code with no current meaning.
# self._validate_no_upsampling(num_feat_patches)
```

Avoid stale TODO comments. If a note is important, make it specific and local.
If it is planning work, track it outside the code.

Avoid decorative comments around simple code blocks. The code should carry the
visual structure through names and whitespace.
