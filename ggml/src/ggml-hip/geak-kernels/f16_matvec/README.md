# f16_matvec

Standalone GEAK workbench for dense decode/prefill matvec rows such as
`mul_mat_vec_f<half,half,1,256>`.

The baseline computes `out = X_fp16 * y_fp16` with float accumulation.

## Input Shapes

`--profile` uses `shape=1`; `--correctness`, `--benchmark`, and
`--full-benchmark` use all shapes.

| Shape | rows | k | outputs |
| --- | ---: | ---: | ---: |
| 0 | 512 | 2048 | 512 |
| 1 | 1024 | 4096 | 1024 |
| 2 | 1536 | 8192 | 1536 |
