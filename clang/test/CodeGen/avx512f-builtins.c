// RUN: %clang_cc1 %s -O0 -triple=x86_64-apple-darwin -target-feature +avx512f -emit-llvm -o - -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

__m512d test_mm512_sqrt_pd(__m512d a)
{
  // CHECK-LABEL: @test_mm512_sqrt_pd
  // CHECK: @llvm.x86.avx512.sqrt.pd.512
  return _mm512_sqrt_pd(a);
}

__m512 test_mm512_sqrt_ps(__m512 a)
{
  // CHECK-LABEL: @test_mm512_sqrt_ps
  // CHECK: @llvm.x86.avx512.sqrt.ps.512
  return _mm512_sqrt_ps(a);
}

__m512d test_mm512_rsqrt14_pd(__m512d a)
{
  // CHECK-LABEL: @test_mm512_rsqrt14_pd
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.512
  return _mm512_rsqrt14_pd(a);
}

__m512 test_mm512_rsqrt14_ps(__m512 a)
{
  // CHECK-LABEL: @test_mm512_rsqrt14_ps
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.512
  return _mm512_rsqrt14_ps(a);
}

__m512 test_mm512_add_ps(__m512 a, __m512 b)
{
  // CHECK-LABEL: @test_mm512_add_ps
  // CHECK: fadd <16 x float>
  return _mm512_add_ps(a, b);
}

__m512d test_mm512_add_pd(__m512d a, __m512d b)
{
  // CHECK-LABEL: @test_mm512_add_pd
  // CHECK: fadd <8 x double>
  return _mm512_add_pd(a, b);
}

__m512 test_mm512_mul_ps(__m512 a, __m512 b)
{
  // CHECK-LABEL: @test_mm512_mul_ps
  // CHECK: fmul <16 x float>
  return _mm512_mul_ps(a, b);
}

__m512d test_mm512_mul_pd(__m512d a, __m512d b)
{
  // CHECK-LABEL: @test_mm512_mul_pd
  // CHECK: fmul <8 x double>
  return _mm512_mul_pd(a, b);
}

void test_mm512_storeu_ps(void *p, __m512 a)
{
  // CHECK-LABEL: @test_mm512_storeu_ps
  // CHECK: @llvm.x86.avx512.mask.storeu.ps.512
  _mm512_storeu_ps(p, a);
}

void test_mm512_storeu_pd(void *p, __m512d a)
{
  // CHECK-LABEL: @test_mm512_storeu_pd
  // CHECK: @llvm.x86.avx512.mask.storeu.pd.512
  _mm512_storeu_pd(p, a);
}

void test_mm512_store_ps(void *p, __m512 a)
{
  // CHECK-LABEL: @test_mm512_store_ps
  // CHECK: store <16 x float>
  _mm512_store_ps(p, a);
}

void test_mm512_store_pd(void *p, __m512d a)
{
  // CHECK-LABEL: @test_mm512_store_pd
  // CHECK: store <8 x double>
  _mm512_store_pd(p, a);
}

__m512 test_mm512_loadu_ps(void *p)
{
  // CHECK-LABEL: @test_mm512_loadu_ps
  // CHECK: load <16 x float>* {{.*}}, align 1{{$}}
  return _mm512_loadu_ps(p);
}

__m512d test_mm512_loadu_pd(void *p)
{
  // CHECK-LABEL: @test_mm512_loadu_pd
  // CHECK: load <8 x double>* {{.*}}, align 1{{$}}
  return _mm512_loadu_pd(p);
}

__m512d test_mm512_set1_pd(double d)
{
  // CHECK-LABEL: @test_mm512_set1_pd
  // CHECK: insertelement <8 x double> {{.*}}, i32 0
  // CHECK: insertelement <8 x double> {{.*}}, i32 1
  // CHECK: insertelement <8 x double> {{.*}}, i32 2
  // CHECK: insertelement <8 x double> {{.*}}, i32 3
  // CHECK: insertelement <8 x double> {{.*}}, i32 4
  // CHECK: insertelement <8 x double> {{.*}}, i32 5
  // CHECK: insertelement <8 x double> {{.*}}, i32 6
  // CHECK: insertelement <8 x double> {{.*}}, i32 7
  return _mm512_set1_pd(d);
}

__m512d test_mm512_castpd256_pd512(__m256d a)
{
  // CHECK-LABEL: @test_mm512_castpd256_pd512
  // CHECK: shufflevector <4 x double> {{.*}} <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  return _mm512_castpd256_pd512(a);
}

__mmask16 test_mm512_knot(__mmask16 a)
{
  // CHECK-LABEL: @test_mm512_knot
  // CHECK: @llvm.x86.avx512.knot.w
  return _mm512_knot(a);
}

__m512i test_mm512_valign_epi64(__m512i a, __m512i b)
{
  // CHECK-LABEL: @test_mm512_valign_epi64
  // CHECK: @llvm.x86.avx512.mask.valign.q.512
  return _mm512_valign_epi64(a, b, 2);
}

__m512d test_mm512_broadcastsd_pd(__m128d a)
{
  // CHECK-LABEL: @test_mm512_broadcastsd_pd
  // CHECK: insertelement <8 x double> {{.*}}, i32 0
  // CHECK: insertelement <8 x double> {{.*}}, i32 1
  // CHECK: insertelement <8 x double> {{.*}}, i32 2
  // CHECK: insertelement <8 x double> {{.*}}, i32 3
  // CHECK: insertelement <8 x double> {{.*}}, i32 4
  // CHECK: insertelement <8 x double> {{.*}}, i32 5
  // CHECK: insertelement <8 x double> {{.*}}, i32 6
  // CHECK: insertelement <8 x double> {{.*}}, i32 7
  return _mm512_broadcastsd_pd(a);
}

__m512i test_mm512_fmadd_pd(__m512d a, __m512d b, __m512d c)
{
  // CHECK-LABEL: @test_mm512_fmadd_pd
  // CHECK: @llvm.x86.fma.mask.vfmadd.pd.512
  return _mm512_fmadd_pd(a, b, c);
}

__mmask16 test_mm512_cmpeq_epi32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpeq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.d.512
  return (__mmask16)_mm512_cmpeq_epi32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmpeq_epi32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpeq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.d.512
  return (__mmask16)_mm512_mask_cmpeq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm512_mask_cmpeq_epi64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpeq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.q.512
  return (__mmask8)_mm512_mask_cmpeq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmpeq_epi64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpeq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.q.512
  return (__mmask8)_mm512_cmpeq_epi64_mask(__a, __b);
}

__m512d test_mm512_unpackhi_pd(__m512d a, __m512d b)
{
  // CHECK-LABEL: @test_mm512_unpackhi_pd
  // CHECK: shufflevector <8 x double> {{.*}} <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  return _mm512_unpackhi_pd(a, b);
}

__m512d test_mm512_unpacklo_pd(__m512d a, __m512d b)
{
  // CHECK-LABEL: @test_mm512_unpacklo_pd
  // CHECK: shufflevector <8 x double> {{.*}} <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  return _mm512_unpacklo_pd(a, b);
}

__m512d test_mm512_unpackhi_ps(__m512d a, __m512d b)
{
  // CHECK-LABEL: @test_mm512_unpackhi_ps
  // CHECK: shufflevector <16 x float> {{.*}} <i32 2, i32 18, i32 3, i32 19, i32 6, i32 22, i32 7, i32 23, i32 10, i32 26, i32 11, i32 27, i32 14, i32 30, i32 15, i32 31>
  return _mm512_unpackhi_ps(a, b);
}

__m512d test_mm512_unpacklo_ps(__m512d a, __m512d b)
{
  // CHECK-LABEL: @test_mm512_unpacklo_ps
  // CHECK: shufflevector <16 x float> {{.*}} <i32 0, i32 16, i32 1, i32 17, i32 4, i32 20, i32 5, i32 21, i32 8, i32 24, i32 9, i32 25, i32 12, i32 28, i32 13, i32 29>
  return _mm512_unpacklo_ps(a, b);
}
