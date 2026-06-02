# dsv4_fp8_kv_quantize

Standalone GEAK workbench for `dsv4_fp8_kv_quantize_f32`.

The baseline mirrors the current DSV4 helper: each row quantizes the non-RoPE
prefix in chunks of 64 using FP8 E4M3FN dequantized values, then copies the
rotary tail unchanged.

## Input Shapes

`n_nope = ne00 - n_rot`; quantization is applied to the non-RoPE prefix.
`--profile` uses `shape=1`; `--correctness`, `--benchmark`, and
`--full-benchmark` use all shapes.

| Shape | rows | ne00 | n_rot | n_nope | elements |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 512 | 96 | 32 | 64 | 49152 |
| 1 | 1024 | 128 | 64 | 64 | 131072 |
| 2 | 1536 | 192 | 64 | 128 | 294912 |
