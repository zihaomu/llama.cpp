# q2_k_mmq

Standalone GEAK workbench for the prefill-side
`mul_mat_q<Q2_K,64,false>` hotspot.

This is the largest current prefill kernel family. The baseline uses direct
Q2_K dequantization and one output element per HIP block.

## Input Shapes

These are prefill-side MMQ shapes with `cols=64`. `--profile` uses `shape=1`;
`--correctness`, `--benchmark`, and `--full-benchmark` use all shapes.

| Shape | rows | cols | k | outputs |
| --- | ---: | ---: | ---: | ---: |
| 0 | 128 | 64 | 2048 | 8192 |
| 1 | 256 | 64 | 4096 | 16384 |
| 2 | 384 | 64 | 8192 | 24576 |
