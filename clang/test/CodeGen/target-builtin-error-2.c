// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -verify -o -
#define __MM_MALLOC_H

#include <x86intrin.h>

// Since we do code generation on a function level this needs to error out since
// the subtarget feature won't be available.
__m256d wombat(__m128i a) {
  if (__builtin_cpu_supports("avx"))
    return __builtin_ia32_cvtdq2pd256((__v4si)a); // expected-error {{'__builtin_ia32_cvtdq2pd256' needs target feature avx}}
  else
    return (__m256d){0, 0, 0, 0};
}
