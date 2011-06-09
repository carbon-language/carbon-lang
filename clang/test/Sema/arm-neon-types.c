// RUN: %clang_cc1 -triple thumbv7-apple-darwin10 -target-cpu cortex-a8 -fsyntax-only -Wvector-conversions -ffreestanding -verify %s

#include <arm_neon.h>

// Radar 8228022: Should not report incompatible vector types.
int32x2_t test(int32x2_t x) {
  return vshr_n_s32(x, 31);
}

// ...but should warn when the types really do not match.
float32x2_t test2(uint32x2_t x) {
  return vcvt_n_f32_s32(x, 9); // expected-warning {{incompatible vector types}}
}

// Check immediate range for vcvt_n intrinsics is 1 to 32.  Radar 9558930.
float32x2_t test3(uint32x2_t x) {
  // FIXME: The "incompatible result type" error is due to pr10112 and should be
  // removed when that is fixed.
  return vcvt_n_f32_u32(x, 0); // expected-error {{argument should be a value from 1 to 32}} expected-error {{incompatible result type}}
}
