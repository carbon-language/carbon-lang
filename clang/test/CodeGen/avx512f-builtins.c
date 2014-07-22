// RUN: %clang_cc1 %s -O0 -triple=x86_64-apple-darwin -target-feature +avx512f -emit-llvm -o - -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

__m512d test_mm512_sqrt_pd(__m512d a)
{
  // CHECK: @llvm.x86.avx512.sqrt.pd.512
  return _mm512_sqrt_pd(a);
}

__m512 test_mm512_sqrt_ps(__m512 a)
{
  // CHECK: @llvm.x86.avx512.sqrt.ps.512
  return _mm512_sqrt_ps(a);
}

__m512d test_mm512_rsqrt14_pd(__m512d a)
{
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.512
  return _mm512_rsqrt14_pd(a);
}

__m512 test_mm512_rsqrt14_ps(__m512 a)
{
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.512
  return _mm512_rsqrt14_ps(a);
}
