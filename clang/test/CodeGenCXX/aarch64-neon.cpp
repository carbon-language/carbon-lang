// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s

// Test whether arm_neon.h can be used in .cpp file.

#include "arm_neon.h"

poly64x1_t test_vld1_p64(poly64_t const * ptr) {
  // CHECK: test_vld1_p64
  return vld1_p64(ptr);
  // CHECK:  ld1 {{{v[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
}
