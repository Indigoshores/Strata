#include "strata/program/lookahead_policy.hpp"

#include <cstdio>

using strata::program::keep_router_lookahead;

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::printf("lookahead_policy_test FAIL line %d: %s\n", __LINE__, #cond); \
            return 1;                                                                  \
        }                                                                              \
        ++checks;                                                                      \
    } while (0)

int main() {
    int checks = 0;

    // Any file-backed expert means the default predictor may still warm a future miss.
    CHECK(keep_router_lookahead(false, false, false, false));

    // The default all-resident path has no file-tier work and stops submitting router dots.
    CHECK(!keep_router_lookahead(true, false, false, false));

    // These are explicit consumers/overrides and must keep the predictor alive.
    CHECK(keep_router_lookahead(true, true, false, false));
    CHECK(keep_router_lookahead(true, false, true, false));
    CHECK(keep_router_lookahead(true, false, false, true));
    CHECK(keep_router_lookahead(true, true, true, true));

    std::printf("lookahead_policy_test OK (%d checks)\n", checks);
}
