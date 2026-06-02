# GEAK Kernel Workbenches For DSV4 HIP

This directory contains standalone HIP workbenches for the DSV4 llama.cpp
hotspot kernels observed in the local ROCm profiles.

Each kernel lives in its own subdirectory with:

- `kernel.hip`: the HIP kernel GEAK should optimize;
- `bench.cpp`: correctness/profile/benchmark harness;
- `COMMANDMENT.md`: the GEAK command contract;
- `README.md`: kernel purpose and current mapping.

The baseline kernels prioritize correctness and a clean optimization surface.
They are not wired into the llama.cpp runtime.

## Input Shape Contract

Kernel performance is shape-dependent. GEAK should optimize against the shapes
below, not a generic or unspecified input size.

- `--correctness`, `--benchmark`, and `--full-benchmark` run all listed shapes.
- `--profile` runs the middle shape only (`shape=1`) to keep profiling focused.
- `cols=1` is decode-side MMVQ; `cols=64` is prefill-side MMQ.

Quantized decode MMVQ shapes for `q8_0_mmvq`, `q2_k_mmvq`, and
`iq2_xxs_mmvq`:

| Shape | rows | cols | k | outputs |
| --- | ---: | ---: | ---: | ---: |
| 0 | 256 | 1 | 2048 | 256 |
| 1 | 512 | 1 | 4096 | 512 |
| 2 | 768 | 1 | 8192 | 768 |

Quantized prefill MMQ shapes for `q8_0_mmq`, `q2_k_mmq`, and `iq2_xxs_mmq`:

| Shape | rows | cols | k | outputs |
| --- | ---: | ---: | ---: | ---: |
| 0 | 128 | 64 | 2048 | 8192 |
| 1 | 256 | 64 | 4096 | 16384 |
| 2 | 384 | 64 | 8192 | 24576 |

Dense FP16 matvec shapes:

| Shape | rows | k | outputs |
| --- | ---: | ---: | ---: |
| 0 | 512 | 2048 | 512 |
| 1 | 1024 | 4096 | 1024 |
| 2 | 1536 | 8192 | 1536 |

DSV4 helper shapes:

| Kernel | Shape | rows | other dimensions |
| --- | ---: | ---: | --- |
| `dsv4_hc_split_sinkhorn` | 0 | 512 | `n_hc=4`, `iters=3`, `width=24` |
| `dsv4_hc_split_sinkhorn` | 1 | 1024 | `n_hc=8`, `iters=2`, `width=80` |
| `dsv4_hc_split_sinkhorn` | 2 | 1536 | `n_hc=16`, `iters=2`, `width=288` |
| `dsv4_fp8_kv_quantize` | 0 | 512 | `ne00=96`, `n_rot=32`, `n_nope=64` |
| `dsv4_fp8_kv_quantize` | 1 | 1024 | `ne00=128`, `n_rot=64`, `n_nope=64` |
| `dsv4_fp8_kv_quantize` | 2 | 1536 | `ne00=192`, `n_rot=64`, `n_nope=128` |
| `dsv4_rope_tail` | 0 | 512 | `ne00=80`, `n_dims=32`, `mode=0`, `n_nope=48` |
| `dsv4_rope_tail` | 1 | 1024 | `ne00=128`, `n_dims=64`, `mode=1`, `n_nope=64` |
| `dsv4_rope_tail` | 2 | 1536 | `ne00=192`, `n_dims=64`, `mode=0`, `n_nope=128` |

Build all workbenches:

```bash
cmake -S ggml/src/ggml-hip/geak-kernels -B build-geak-kernels \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-7.12/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm/core-7.12 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build-geak-kernels -j
```

Run one workbench:

```bash
./build-geak-kernels/geak_q8_0_mmvq --correctness
./build-geak-kernels/geak_q8_0_mmvq --benchmark --iterations 200
```

GEAK example:

```bash
geak --repo /home/zihao/zihao_workspace/speed_up_dsv4/ggml/src/ggml-hip/geak-kernels \
  --kernel-url q8_0_mmvq/kernel.hip \
  --test-command "cmake -S . -B build -DCMAKE_HIP_COMPILER=/opt/rocm/core-7.12/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm/core-7.12 -DCMAKE_HIP_ARCHITECTURES=gfx1151 && cmake --build build -j && ./build/geak_q8_0_mmvq --correctness && ./build/geak_q8_0_mmvq --benchmark --iterations 200" \
  --task "Optimize q8_0_mmvq. Metric: median kernel latency in ms, lower is better."
```

## Included Kernels

| Directory | Profile kernel mapped |
| --- | --- |
| `q8_0_mmvq` | `mul_mat_vec_q<Q8_0,1,false,false>` |
| `q2_k_mmvq` | `mul_mat_vec_q<Q2_K,1,false,false>` |
| `iq2_xxs_mmvq` | `mul_mat_vec_q<IQ2_XXS,1,false,false>` |
| `q8_0_mmq` | `mul_mat_q<Q8_0,64,false>` |
| `q2_k_mmq` | `mul_mat_q<Q2_K,64,false>` |
| `iq2_xxs_mmq` | `mul_mat_q<IQ2_XXS,64,false>` |
| `f16_matvec` | `mul_mat_vec_f<half,half,1,...>` |
| `dsv4_hc_split_sinkhorn` | `dsv4_hc_split_sinkhorn_f32` |
| `dsv4_fp8_kv_quantize` | `dsv4_fp8_kv_quantize_f32` |
| `dsv4_rope_tail` | `dsv4_rope_tail` |

The strict Step39 Top10 also contains rocBLAS/Tensile GEMM rows and copy
kernels. Those are intentionally excluded because they are library calls or
layout/copy symptoms rather than useful standalone native-HIP kernel targets.
