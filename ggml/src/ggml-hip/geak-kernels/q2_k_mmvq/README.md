# q2_k_mmvq

Standalone GEAK workbench for the decode-side
`mul_mat_vec_q<Q2_K,1,false,false>` hotspot.

The baseline uses the llama.cpp Q2_K block layout and computes a direct
dequantized dot product against Q8_1 activations.
