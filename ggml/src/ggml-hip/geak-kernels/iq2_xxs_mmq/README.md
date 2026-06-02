# iq2_xxs_mmq

Standalone GEAK workbench for the prefill-side
`mul_mat_q<IQ2_XXS,64,false>` hotspot.

The baseline uses llama.cpp IQ2_XXS codebook tables and computes one output
element per HIP block.

## Input Shapes

These are prefill-side MMQ shapes with `cols=64`. `--profile` uses `shape=1`;
`--correctness`, `--benchmark`, and `--full-benchmark` use all shapes.

| Shape | rows | cols | k | outputs |
| --- | ---: | ---: | ---: | ---: |
| 0 | 128 | 64 | 2048 | 8192 |
| 1 | 256 | 64 | 4096 | 16384 |
| 2 | 384 | 64 | 8192 | 24576 |
