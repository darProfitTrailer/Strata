#pragma once
namespace strata::kernels::cpu {
enum class NativeCpuIsaFloor { Avx2, Avx, Sse42 };
// Pure predicate: callers supply OS-aware probes, never an override that enables instructions.
constexpr bool native_cpu_floor_supported(NativeCpuIsaFloor required, bool avx2, bool avx, bool sse42) {
    switch (required) {
        case NativeCpuIsaFloor::Avx2: return avx2;
        case NativeCpuIsaFloor::Avx: return avx;
        case NativeCpuIsaFloor::Sse42: return sse42;
    }
    return false;
}
} // namespace strata::kernels::cpu
