# COMMANDMENT dsv4_fp8_kv_quantize

## SETUP

```bash
cmake -S . -B build -DCMAKE_HIP_COMPILER=/opt/rocm/core-7.12/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm/core-7.12 -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build -j
```

## CORRECTNESS

```bash
./build/geak_dsv4_fp8_kv_quantize --correctness
```

## PROFILE

```bash
./build/geak_dsv4_fp8_kv_quantize --profile --iterations 50
```

## BENCHMARK

```bash
./build/geak_dsv4_fp8_kv_quantize --benchmark --iterations 200
```
