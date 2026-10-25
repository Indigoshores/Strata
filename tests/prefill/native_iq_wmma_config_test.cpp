// The gfx1151 exception to clang 22's conservative WMMA occupancy cap is host-testable: no model or GPU.
#include "native_iq_wmma_config.hpp"

#include <cstdio>

using strata::prefill::fused::config::native_iq_wmma_waves_per_eu;

static_assert(native_iq_wmma_waves_per_eu(21, false) == 8, "older clang keeps the measured occupancy cap");
static_assert(native_iq_wmma_waves_per_eu(22, false) == 1, "clang 22 remains conservative off gfx1151");
static_assert(native_iq_wmma_waves_per_eu(23, false) == 1, "later clang remains conservative off gfx1151");
static_assert(native_iq_wmma_waves_per_eu(22, true) == 8, "gfx1151 restores the measured occupancy cap");
static_assert(native_iq_wmma_waves_per_eu(23, true) == 8, "gfx1151 exception persists on later clang");

int main() {
    std::printf("native_iq_wmma_config_test: ok\n");
    return 0;
}
