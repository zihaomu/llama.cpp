# f16_matvec

Standalone GEAK workbench for dense decode/prefill matvec rows such as
`mul_mat_vec_f<half,half,1,256>`.

The baseline computes `out = X_fp16 * y_fp16` with float accumulation.
