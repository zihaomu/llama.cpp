#include "../geak_common.hpp"

extern "C" void geak_q8_0_mmq_launch(
        const block_q8_0 *, const block_q8_1 *, float *,
        int, int, int, hipStream_t);

int main(int argc, char ** argv) {
    return geak_run_q8_0_matmul("q8_0_mmq", 64, geak_q8_0_mmq_launch, argc, argv);
}
