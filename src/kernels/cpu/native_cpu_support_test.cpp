#include "strata/kernels/cpu/native_cpu_support.hpp"
using namespace strata::kernels::cpu;
static_assert(!native_cpu_floor_supported(NativeCpuIsaFloor::Avx2, false, false, true));
static_assert(!native_cpu_floor_supported(NativeCpuIsaFloor::Avx, false, false, true));
static_assert(native_cpu_floor_supported(NativeCpuIsaFloor::Sse42, false, false, true));
static_assert(!native_cpu_floor_supported(NativeCpuIsaFloor::Sse42, false, false, false));
static_assert(native_cpu_floor_supported(NativeCpuIsaFloor::Avx, false, true, true));
static_assert(!native_cpu_floor_supported(NativeCpuIsaFloor::Avx2, false, true, true));
static_assert(native_cpu_floor_supported(NativeCpuIsaFloor::Avx2, true, true, true));
int main() { return 0; }
