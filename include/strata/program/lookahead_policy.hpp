#pragma once

namespace strata::program {

// Once the final cache residency table is available, the routing predictor has no work when every expert is
// already on the GPU.  Measurement, Foresight and an explicit STRATA_LOOKAHEAD=1 remain opt-ins: each needs the
// predictions even though the ordinary file-tier warmer does not.
inline bool keep_router_lookahead(bool all_experts_resident, bool explicitly_enabled, bool stats,
                                  bool foresight) noexcept {
    return !all_experts_resident || explicitly_enabled || stats || foresight;
}

}  // namespace strata::program
