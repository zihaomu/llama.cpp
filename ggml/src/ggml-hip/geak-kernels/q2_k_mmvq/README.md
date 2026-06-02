# q2_k_mmvq

Standalone GEAK workbench for the decode-side
`mul_mat_vec_q<Q2_K,1,false,false>` hotspot.

The baseline uses the llama.cpp Q2_K block layout and computes a direct
dequantized dot product against Q8_1 activations.

## Input Shapes

These are decode-side MMVQ shapes with `cols=1`. `--profile` uses `shape=1`;
`--correctness`, `--benchmark`, and `--full-benchmark` use all shapes.

| Shape | rows | cols | k | outputs |
| --- | ---: | ---: | ---: | ---: |
| 0 | 256 | 1 | 2048 | 256 |
| 1 | 512 | 1 | 4096 | 512 |
| 2 | 768 | 1 | 8192 | 768 |
