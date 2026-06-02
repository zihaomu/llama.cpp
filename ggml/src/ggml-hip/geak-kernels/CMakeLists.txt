cmake_minimum_required(VERSION 3.21)

project(dsv4_geak_hip_kernels LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

if(NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx1151)
endif()

set(GEAK_COMMON_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    /opt/rocm/core-7.12/include
)

function(add_geak_kernel target dir)
    add_executable(${target}
        ${dir}/bench.cpp
        ${dir}/kernel.hip
    )
    target_include_directories(${target} PRIVATE ${GEAK_COMMON_INCLUDES})
    target_compile_definitions(${target} PRIVATE GGML_USE_HIP __HIP_PLATFORM_AMD__)
    target_link_directories(${target} PRIVATE /opt/rocm/core-7.12/lib)
    set_target_properties(${target} PROPERTIES
        BUILD_RPATH "/opt/rocm/core-7.12/lib"
    )
endfunction()

add_geak_kernel(geak_q8_0_mmvq q8_0_mmvq)
add_geak_kernel(geak_q2_k_mmvq q2_k_mmvq)
add_geak_kernel(geak_iq2_xxs_mmvq iq2_xxs_mmvq)
add_geak_kernel(geak_q8_0_mmq q8_0_mmq)
add_geak_kernel(geak_q2_k_mmq q2_k_mmq)
add_geak_kernel(geak_iq2_xxs_mmq iq2_xxs_mmq)
add_geak_kernel(geak_f16_matvec f16_matvec)
add_geak_kernel(geak_dsv4_hc_split_sinkhorn dsv4_hc_split_sinkhorn)
add_geak_kernel(geak_dsv4_fp8_kv_quantize dsv4_fp8_kv_quantize)
add_geak_kernel(geak_dsv4_rope_tail dsv4_rope_tail)
