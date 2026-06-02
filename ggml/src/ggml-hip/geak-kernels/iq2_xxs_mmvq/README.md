# iq2_xxs_mmvq

Standalone GEAK workbench for the decode-side
`mul_mat_vec_q<IQ2_XXS,1,false,false>` hotspot.

The baseline uses llama.cpp IQ2_XXS codebook tables from `ggml-common.h`.

## Input Shapes

These are decode-side MMVQ shapes with `cols=1`. `--profile` uses `shape=1`;
`--correctness`, `--benchmark`, and `--full-benchmark` use all shapes.

| Shape | rows | cols | k | outputs |
| --- | ---: | ---: | ---: | ---: |
| 0 | 256 | 1 | 2048 | 256 |
| 1 | 512 | 1 | 4096 | 512 |
| 2 | 768 | 1 | 8192 | 768 |
