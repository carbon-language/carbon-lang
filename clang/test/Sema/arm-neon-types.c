// RUN: %clang_cc1 -triple thumbv7-apple-darwin10 -target-cpu cortex-a8 -fsyntax-only -Wvector-conversions -verify %s

#include <arm_neon.h>

// Radar 8228022: Should not report incompatible vector types.
int32x2_t test(int32x2_t x) {
  return vshr_n_s32(x, 31);
}

// ...but should warn when the types really do not match.
float32x2_t test2(uint32x2_t x) {
  return vcvt_n_f32_s32(x, 0); // expected-warning {{incompatible vector types}}
}
