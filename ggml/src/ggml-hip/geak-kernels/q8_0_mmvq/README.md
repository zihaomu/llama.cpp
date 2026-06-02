# q8_0_mmvq

Standalone GEAK workbench for the decode-side
`mul_mat_vec_q<Q8_0,1,false,false>` hotspot.

The baseline computes one output row per HIP block using direct Q8_0 and Q8_1
dequantization plus a block reduction. It is correctness-first and intentionally
leaves row tiling, Q8_1 reuse, DP4A packing, and LDS staging for GEAK.

## Input Shapes

These are decode-side MMVQ shapes with `cols=1`. `--profile` uses `shape=1`;
`--correctness`, `--benchmark`, and `--full-benchmark` use all shapes.

| Shape | rows | cols | k | outputs |
| --- | ---: | ---: | ---: | ---: |
| 0 | 256 | 1 | 2048 | 256 |
| 1 | 512 | 1 | 4096 | 512 |
| 2 | 768 | 1 | 8192 | 768 |
