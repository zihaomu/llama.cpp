# q8_0_mmq

Standalone GEAK workbench for the prefill-side
`mul_mat_q<Q8_0,64,false>` hotspot.

The baseline computes a 64-column prompt batch with direct Q8_0/Q8_1
dequantization and one output element per HIP block.
