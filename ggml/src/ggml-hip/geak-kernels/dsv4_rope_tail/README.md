# dsv4_rope_tail

Standalone GEAK workbench for `dsv4_rope_tail`.

The baseline covers the normal and NeoX tail rotation modes with contiguous
F32 input. It keeps YaRN/frequency-factor integration out of scope for this
first GEAK harness.

## Input Shapes

`n_nope = ne00 - n_dims`; only the tail dimensions are rotated. `mode=0` is
normal RoPE and `mode=1` is NeoX. `--profile` uses `shape=1`; `--correctness`,
`--benchmark`, and `--full-benchmark` use all shapes.

| Shape | rows | ne00 | n_dims | mode | n_nope | elements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 512 | 80 | 32 | 0 | 48 | 40960 |
| 1 | 1024 | 128 | 64 | 1 | 64 | 131072 |
| 2 | 1536 | 192 | 64 | 0 | 128 | 294912 |
