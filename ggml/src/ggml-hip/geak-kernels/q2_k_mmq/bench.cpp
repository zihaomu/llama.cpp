#include "../geak_common.hpp"

extern "C" void geak_q2_k_mmq_launch(
        const block_q2_K *, const block_q8_1 *, float *,
        int, int, int, hipStream_t);

int main(int argc, char ** argv) {
    return geak_run_q2_k_matmul("q2_k_mmq", 64, geak_q2_k_mmq_launch, argc, argv);
}
