// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -o -
#define __MM_MALLOC_H

#include <x86intrin.h>

// No warnings.
extern __m256i a;
int __attribute__((target("avx"))) bar(__m256i a) {
  return _mm256_extract_epi32(a, 3);
}

int baz() {
  return bar(a);
}

int __attribute__((target("avx"))) qq_avx(__m256i a) {
  return _mm256_extract_epi32(a, 3);
}

int qq_noavx() {
  return 0;
}

extern __m256i a;
int qq() {
  if (__builtin_cpu_supports("avx"))
    return qq_avx(a);
  else
    return qq_noavx();
}
