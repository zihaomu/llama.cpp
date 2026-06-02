# COMMANDMENT f16_matvec

## SETUP

```bash
cmake -S . -B build -DCMAKE_HIP_COMPILER=/opt/rocm/core-7.12/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm/core-7.12 -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build -j
```

## CORRECTNESS

```bash
./build/geak_f16_matvec --correctness
```

## PROFILE

```bash
./build/geak_f16_matvec --profile --iterations 50
```

## BENCHMARK

```bash
./build/geak_f16_matvec --benchmark --iterations 200
```
