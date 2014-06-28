// RUN: %clang_cc1 -triple armv8 -target-cpu cortex-a57 -fsyntax-only -ffreestanding -verify %s

#include <arm_acle.h>

/*
 * Saturating intrinsics
 * Second argument for SSAT and USAT intrinsics must be compile-time constant,
 * otherwise an error should be raised.
 */
int32_t test_ssat_const_diag(int32_t t, const int32_t v) {
  return __ssat(t, v);  // expected-error-re {{argument to {{.*}} must be a constant integer}}
}

int32_t test_usat_const_diag(int32_t t, const int32_t v) {
  return __usat(t, v);  // expected-error-re {{argument to {{.*}} must be a constant integer}}
}
