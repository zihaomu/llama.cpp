# dsv4_hc_split_sinkhorn

Standalone GEAK workbench for `dsv4_hc_split_sinkhorn_f32`.

This baseline mirrors the current llama.cpp HIP/CUDA helper kernel with
contiguous row-major inputs.

## Input Shapes

`width = (2 + n_hc) * n_hc`. `--profile` uses `shape=1`; `--correctness`,
`--benchmark`, and `--full-benchmark` use all shapes.

| Shape | rows | n_hc | iters | width | outputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 512 | 4 | 3 | 24 | 12288 |
| 1 | 1024 | 8 | 2 | 80 | 81920 |
| 2 | 1536 | 16 | 2 | 288 | 442368 |
