#include "../geak_common.hpp"

extern "C" void geak_iq2_xxs_mmq_launch(
        const block_iq2_xxs *, const block_q8_1 *, float *,
        int, int, int, hipStream_t);

int main(int argc, char ** argv) {
    return geak_run_iq2_xxs_matmul("iq2_xxs_mmq", 64, geak_iq2_xxs_mmq_launch, argc, argv);
}
