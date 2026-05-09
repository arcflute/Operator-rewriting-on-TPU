# Operator Rewriting on TPU

This repository records my TileLang-TPU rewriting work for the `mm_w8a16_dq` operator.

## Target operator

The target operator is `mm_w8a16_dq`.

Core computation:

right_fp8 -> right_fp16
right_scaled = right_fp16 * scale
out = left @ right_scaled.T

## Main work

1. Added FP8 dtype support for TileLang-TPU code generation.
2. Implemented FP8 to FP16 conversion through TileLang `T.ppl_copy`.
3. Implemented dequantization by multiplying FP16-converted weights with scale.
4. Implemented fixed-shape TileLang functional-equivalent versions for the original PPL kernel paths.

## Implemented paths

- path_mn_mn
- path_mn_k
- path_k_k_n64
- path_k_mn_n64

## Notes

- Current validation is based on cmodel correctness.
- cmodel execution time does not represent real TPU performance.
- `block_N=64` is used as the safe version because `block_N=128` may trigger an NPU broadcast lane limitation.
