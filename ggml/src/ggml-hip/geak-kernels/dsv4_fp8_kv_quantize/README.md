# dsv4_fp8_kv_quantize

Standalone GEAK workbench for `dsv4_fp8_kv_quantize_f32`.

The baseline mirrors the current DSV4 helper: each row quantizes the non-RoPE
prefix in chunks of 64 using FP8 E4M3FN dequantized values, then copies the
rotary tail unchanged.
