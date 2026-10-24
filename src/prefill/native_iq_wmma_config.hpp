// Compile-time policy for the native IQ prompt-expert WMMA kernel's VGPR occupancy cap.
#pragma once

namespace strata::prefill::fused::config {

constexpr int native_iq_wmma_waves_per_eu(int clang_major, bool gfx1151) {
    // clang 22 miscompiled the capped kernel on gfx1100 (#1180).  gfx1151 has been
    // measured correct and faster with the original eight-wave cap.
    return gfx1151 || clang_major < 22 ? 8 : 1;
}

}  // namespace strata::prefill::fused::config
