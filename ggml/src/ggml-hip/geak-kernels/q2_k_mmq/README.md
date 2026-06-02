# q2_k_mmq

Standalone GEAK workbench for the prefill-side
`mul_mat_q<Q2_K,64,false>` hotspot.

This is the largest current prefill kernel family. The baseline uses direct
Q2_K dequantization and one output element per HIP block.
