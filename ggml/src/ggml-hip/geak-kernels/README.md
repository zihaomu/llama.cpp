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
