// RUN: %clang_cc1 %s -O0 -triple=x86_64-apple-darwin -ffreestanding -target-feature +avx512f -target-feature +avx512vl -emit-llvm -o - -Werror | FileCheck %s

#include <immintrin.h>

__mmask8 test_mm256_cmpeq_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.d.256
  return (__mmask8)_mm256_cmpeq_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpeq_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.d.256
  return (__mmask8)_mm256_mask_cmpeq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.d.128
  return (__mmask8)_mm_cmpeq_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.d.128
  return (__mmask8)_mm_mask_cmpeq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpeq_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.q.256
  return (__mmask8)_mm256_cmpeq_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpeq_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.q.256
  return (__mmask8)_mm256_mask_cmpeq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.q.128
  return (__mmask8)_mm_cmpeq_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.q.128
  return (__mmask8)_mm_mask_cmpeq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpgt_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.d.256
  return (__mmask8)_mm256_cmpgt_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.d.256
  return (__mmask8)_mm256_mask_cmpgt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.d.128
  return (__mmask8)_mm_cmpgt_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.d.128
  return (__mmask8)_mm_mask_cmpgt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpgt_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.q.256
  return (__mmask8)_mm256_cmpgt_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.q.256
  return (__mmask8)_mm256_mask_cmpgt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.q.128
  return (__mmask8)_mm_cmpgt_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.q.128
  return (__mmask8)_mm_mask_cmpgt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 0, i8 -1)
  return (__mmask8)_mm_cmpeq_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 0, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpeq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 0, i8 -1)
  return (__mmask8)_mm_cmpeq_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 0, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpeq_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm_cmpge_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpge_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm_cmpge_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpge_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm256_cmpge_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpge_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm256_cmpge_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpge_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm_cmpge_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpge_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm_cmpge_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpge_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm256_cmpge_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpge_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm256_cmpge_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpge_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 6, i8 -1)
  return (__mmask8)_mm_cmpgt_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 6, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpgt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 6, i8 -1)
  return (__mmask8)_mm_cmpgt_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 6, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpgt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpgt_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 6, i8 -1)
  return (__mmask8)_mm256_cmpgt_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 6, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpgt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpgt_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 6, i8 -1)
  return (__mmask8)_mm256_cmpgt_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 6, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpgt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm_cmple_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm_mask_cmple_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm_cmple_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm_mask_cmple_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm256_cmple_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmple_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm256_cmple_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmple_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm_cmple_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm_mask_cmple_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm_cmple_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm_mask_cmple_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm256_cmple_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmple_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm256_cmple_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmple_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm_cmplt_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm_mask_cmplt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm_cmplt_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm_mask_cmplt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm256_cmplt_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmplt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm256_cmplt_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmplt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm_cmplt_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm_mask_cmplt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm_cmplt_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm_mask_cmplt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm256_cmplt_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmplt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm256_cmplt_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmplt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm_cmpneq_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpneq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm_cmpneq_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpneq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm256_cmpneq_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpneq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm256_cmpneq_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpneq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm_cmpneq_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpneq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm_cmpneq_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpneq_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm256_cmpneq_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpneq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm256_cmpneq_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpneq_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmp_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 7, i8 -1)
  return (__mmask8)_mm_cmp_epi32_mask(__a, __b, 7);
}

__mmask8 test_mm_mask_cmp_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask8)_mm_mask_cmp_epi32_mask(__u, __a, __b, 7);
}

__mmask8 test_mm_cmp_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 7, i8 -1)
  return (__mmask8)_mm_cmp_epi64_mask(__a, __b, 7);
}

__mmask8 test_mm_mask_cmp_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask8)_mm_mask_cmp_epi64_mask(__u, __a, __b, 7);
}

__mmask8 test_mm256_cmp_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 7, i8 -1)
  return (__mmask8)_mm256_cmp_epi32_mask(__a, __b, 7);
}

__mmask8 test_mm256_mask_cmp_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmp_epi32_mask(__u, __a, __b, 7);
}

__mmask8 test_mm256_cmp_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 7, i8 -1)
  return (__mmask8)_mm256_cmp_epi64_mask(__a, __b, 7);
}

__mmask8 test_mm256_mask_cmp_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmp_epi64_mask(__u, __a, __b, 7);
}

__mmask8 test_mm_cmp_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 7, i8 -1)
  return (__mmask8)_mm_cmp_epu32_mask(__a, __b, 7);
}

__mmask8 test_mm_mask_cmp_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask8)_mm_mask_cmp_epu32_mask(__u, __a, __b, 7);
}

__mmask8 test_mm_cmp_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 7, i8 -1)
  return (__mmask8)_mm_cmp_epu64_mask(__a, __b, 7);
}

__mmask8 test_mm_mask_cmp_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask8)_mm_mask_cmp_epu64_mask(__u, __a, __b, 7);
}

__mmask8 test_mm256_cmp_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 7, i8 -1)
  return (__mmask8)_mm256_cmp_epu32_mask(__a, __b, 7);
}

__mmask8 test_mm256_mask_cmp_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmp_epu32_mask(__u, __a, __b, 7);
}

__mmask8 test_mm256_cmp_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 7, i8 -1)
  return (__mmask8)_mm256_cmp_epu64_mask(__a, __b, 7);
}

__mmask8 test_mm256_mask_cmp_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmp_epu64_mask(__u, __a, __b, 7);
}

__m512i test_mm512_maskz_andnot_epi32 (__mmask16 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_andnot_epi32
  //CHECK: @llvm.x86.avx512.mask.pandn.d.512
  return _mm512_maskz_andnot_epi32(__k,__A,__B);
}

__m512i test_mm512_mask_andnot_epi32 (__mmask16 __k,__m512i __A, __m512i __B, __m512i __src) {
  //CHECK-LABEL: @test_mm512_mask_andnot_epi32
  //CHECK: @llvm.x86.avx512.mask.pandn.d.512
  return _mm512_mask_andnot_epi32(__src,__k,__A,__B);
}

__m512i test_mm512_andnot_epi32(__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_andnot_epi32
  //CHECK: @llvm.x86.avx512.mask.pandn.d.512
  return _mm512_andnot_epi32(__A,__B);
}

__m512i test_mm512_maskz_andnot_epi64 (__mmask8 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_andnot_epi64
  //CHECK: @llvm.x86.avx512.mask.pandn.q.512
  return _mm512_maskz_andnot_epi64(__k,__A,__B);
}

__m512i test_mm512_mask_andnot_epi64 (__mmask8 __k,__m512i __A, __m512i __B, __m512i __src) {
  //CHECK-LABEL: @test_mm512_mask_andnot_epi64
  //CHECK: @llvm.x86.avx512.mask.pandn.q.512
  return _mm512_mask_andnot_epi64(__src,__k,__A,__B);
}

__m512i test_mm512_andnot_epi64(__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_andnot_epi64
  //CHECK: @llvm.x86.avx512.mask.pandn.q.512
  return _mm512_andnot_epi64(__A,__B);
}

__m256i test_mm256_mask_add_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_add_epi32
  //CHECK: @llvm.x86.avx512.mask.padd.d.256
  return _mm256_mask_add_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_add_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_add_epi32
  //CHECK: @llvm.x86.avx512.mask.padd.d.256
  return _mm256_maskz_add_epi32(__U, __A, __B);
}

__m256i test_mm256_mask_add_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_add_epi64
  //CHECK: @llvm.x86.avx512.mask.padd.q.256
  return _mm256_mask_add_epi64(__W,__U,__A,__B);
}

__m256i test_mm256_maskz_add_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_add_epi64
  //CHECK: @llvm.x86.avx512.mask.padd.q.256
  return _mm256_maskz_add_epi64 (__U,__A,__B);
}

__m256i test_mm256_mask_sub_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_sub_epi32
  //CHECK: @llvm.x86.avx512.mask.psub.d.256
  return _mm256_mask_sub_epi32 (__W,__U,__A,__B);
}

__m256i test_mm256_maskz_sub_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_sub_epi32
  //CHECK: @llvm.x86.avx512.mask.psub.d.256
  return _mm256_maskz_sub_epi32 (__U,__A,__B);
}

__m256i test_mm256_mask_sub_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_sub_epi64
  //CHECK: @llvm.x86.avx512.mask.psub.q.256
  return _mm256_mask_sub_epi64 (__W,__U,__A,__B);
}

__m256i test_mm256_maskz_sub_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_sub_epi64
  //CHECK: @llvm.x86.avx512.mask.psub.q.256
  return _mm256_maskz_sub_epi64 (__U,__A,__B);
}

__m128i test_mm_mask_add_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_add_epi32
  //CHECK: @llvm.x86.avx512.mask.padd.d.128
  return _mm_mask_add_epi32(__W,__U,__A,__B);
}


__m128i test_mm_maskz_add_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_add_epi32
  //CHECK: @llvm.x86.avx512.mask.padd.d.128
  return _mm_maskz_add_epi32 (__U,__A,__B);
}

__m128i test_mm_mask_add_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
//CHECK-LABEL: @test_mm_mask_add_epi64
  //CHECK: @llvm.x86.avx512.mask.padd.q.128
  return _mm_mask_add_epi64 (__W,__U,__A,__B);
}

__m128i test_mm_maskz_add_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_add_epi64
  //CHECK: @llvm.x86.avx512.mask.padd.q.128
  return _mm_maskz_add_epi64 (__U,__A,__B);
}

__m128i test_mm_mask_sub_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_sub_epi32
  //CHECK: @llvm.x86.avx512.mask.psub.d.128
  return _mm_mask_sub_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_sub_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_sub_epi32
  //CHECK: @llvm.x86.avx512.mask.psub.d.128
  return _mm_maskz_sub_epi32(__U, __A, __B);
}

__m128i test_mm_mask_sub_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_sub_epi64
  //CHECK: @llvm.x86.avx512.mask.psub.q.128
  return _mm_mask_sub_epi64 (__W, __U, __A, __B);
}

__m128i test_mm_maskz_sub_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_sub_epi64
  //CHECK: @llvm.x86.avx512.mask.psub.q.128
  return _mm_maskz_sub_epi64 (__U, __A, __B);
}

__m256i test_mm256_mask_mul_epi32 (__m256i __W, __mmask8 __M, __m256i __X,
           __m256i __Y) {
  //CHECK-LABEL: @test_mm256_mask_mul_epi32
  //CHECK: @llvm.x86.avx512.mask.pmul.dq.256
  return _mm256_mask_mul_epi32(__W, __M, __X, __Y);
}

__m256i test_mm256_maskz_mul_epi32 (__mmask8 __M, __m256i __X, __m256i __Y) {
  //CHECK-LABEL: @test_mm256_maskz_mul_epi32
  //CHECK: @llvm.x86.avx512.mask.pmul.dq.256
  return _mm256_maskz_mul_epi32(__M, __X, __Y);
}


__m128i test_mm_mask_mul_epi32 (__m128i __W, __mmask8 __M, __m128i __X,
        __m128i __Y) {
  //CHECK-LABEL: @test_mm_mask_mul_epi32
  //CHECK: @llvm.x86.avx512.mask.pmul.dq.128
  return _mm_mask_mul_epi32(__W, __M, __X, __Y);
}

__m128i test_mm_maskz_mul_epi32 (__mmask8 __M, __m128i __X, __m128i __Y) {
  //CHECK-LABEL: @test_mm_maskz_mul_epi32
  //CHECK: @llvm.x86.avx512.mask.pmul.dq.128
  return _mm_maskz_mul_epi32(__M, __X, __Y);
}

__m256i test_mm256_mask_mul_epu32 (__m256i __W, __mmask8 __M, __m256i __X,
           __m256i __Y) {
  //CHECK-LABEL: @test_mm256_mask_mul_epu32
  //CHECK: @llvm.x86.avx512.mask.pmulu.dq.256
  return _mm256_mask_mul_epu32(__W, __M, __X, __Y);
}

__m256i test_mm256_maskz_mul_epu32 (__mmask8 __M, __m256i __X, __m256i __Y) {
  //CHECK-LABEL: @test_mm256_maskz_mul_epu32
  //CHECK: @llvm.x86.avx512.mask.pmulu.dq.256
  return _mm256_maskz_mul_epu32(__M, __X, __Y);
}

__m128i test_mm_mask_mul_epu32 (__m128i __W, __mmask8 __M, __m128i __X,
        __m128i __Y) {
  //CHECK-LABEL: @test_mm_mask_mul_epu32
  //CHECK: @llvm.x86.avx512.mask.pmulu.dq.128
  return _mm_mask_mul_epu32(__W, __M, __X, __Y);
}

__m128i test_mm_maskz_mul_epu32 (__mmask8 __M, __m128i __X, __m128i __Y) {
  //CHECK-LABEL: @test_mm_maskz_mul_epu32
  //CHECK: @llvm.x86.avx512.mask.pmulu.dq.128
  return _mm_maskz_mul_epu32(__M, __X, __Y);
}

__m128i test_mm_maskz_mullo_epi32 (__mmask8 __M, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_mullo_epi32
  //CHECK: @llvm.x86.avx512.mask.pmull.d.128
  return _mm_maskz_mullo_epi32(__M, __A, __B);
}

__m128i test_mm_mask_mullo_epi32 (__m128i __W, __mmask8 __M, __m128i __A,
          __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_mullo_epi32
  //CHECK: @llvm.x86.avx512.mask.pmull.d.128
  return _mm_mask_mullo_epi32(__W, __M, __A, __B);
}

__m256i test_mm256_maskz_mullo_epi32 (__mmask8 __M, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_mullo_epi32
  //CHECK: @llvm.x86.avx512.mask.pmull.d.256
  return _mm256_maskz_mullo_epi32(__M, __A, __B);
}

__m256i test_mm256_mask_mullo_epi32 (__m256i __W, __mmask8 __M, __m256i __A,
       __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_mullo_epi32
  //CHECK: @llvm.x86.avx512.mask.pmull.d.256
  return _mm256_mask_mullo_epi32(__W, __M, __A, __B);
}

__m256i test_mm256_mask_and_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_and_epi32
  //CHECK: @llvm.x86.avx512.mask.pand.d.256
  return _mm256_mask_and_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_and_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_and_epi32
  //CHECK: @llvm.x86.avx512.mask.pand.d.256
  return _mm256_maskz_and_epi32(__U, __A, __B);
}

__m128i test_mm_mask_and_epi32 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_and_epi32
  //CHECK: @llvm.x86.avx512.mask.pand.d.128
  return _mm_mask_and_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_and_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_and_epi32
  //CHECK: @llvm.x86.avx512.mask.pand.d.128
  return _mm_maskz_and_epi32(__U, __A, __B);
}

__m256i test_mm256_mask_andnot_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
        __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_andnot_epi32
  //CHECK: @llvm.x86.avx512.mask.pandn.d.256
  return _mm256_mask_andnot_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_andnot_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_andnot_epi32
  //CHECK: @llvm.x86.avx512.mask.pandn.d.256
  return _mm256_maskz_andnot_epi32(__U, __A, __B);
}

__m128i test_mm_mask_andnot_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
           __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_andnot_epi32
  //CHECK: @llvm.x86.avx512.mask.pandn.d.128
  return _mm_mask_andnot_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_andnot_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_andnot_epi32
  //CHECK: @llvm.x86.avx512.mask.pandn.d.128
  return _mm_maskz_andnot_epi32(__U, __A, __B);
}

__m256i test_mm256_mask_or_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
          __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_or_epi32
  //CHECK: @llvm.x86.avx512.mask.por.d.256
  return _mm256_mask_or_epi32(__W, __U, __A, __B);
}

 __m256i test_mm256_maskz_or_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_or_epi32
  //CHECK: @llvm.x86.avx512.mask.por.d.256
  return _mm256_maskz_or_epi32(__U, __A, __B);
}

 __m128i test_mm_mask_or_epi32 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_or_epi32
  //CHECK: @llvm.x86.avx512.mask.por.d.128
  return _mm_mask_or_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_or_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_or_epi32
  //CHECK: @llvm.x86.avx512.mask.por.d.128
  return _mm_maskz_or_epi32(__U, __A, __B);
}

__m256i test_mm256_mask_xor_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_xor_epi32
  //CHECK: @llvm.x86.avx512.mask.pxor.d.256
  return _mm256_mask_xor_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_xor_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_xor_epi32
  //CHECK: @llvm.x86.avx512.mask.pxor.d.256
  return _mm256_maskz_xor_epi32(__U, __A, __B);
}

__m128i test_mm_mask_xor_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_xor_epi32
  //CHECK: @llvm.x86.avx512.mask.pxor.d.128
  return _mm_mask_xor_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_xor_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_xor_epi32
  //CHECK: @llvm.x86.avx512.mask.pxor.d.128
  return _mm_maskz_xor_epi32(__U, __A, __B);
}

__m256i test_mm256_mask_and_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_and_epi64
  //CHECK: @llvm.x86.avx512.mask.pand.q.256
  return _mm256_mask_and_epi64(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_and_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_and_epi64
  //CHECK: @llvm.x86.avx512.mask.pand.q.256
  return _mm256_maskz_and_epi64(__U, __A, __B);
}

__m128i test_mm_mask_and_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_and_epi64
  //CHECK: @llvm.x86.avx512.mask.pand.q.128
  return _mm_mask_and_epi64(__W,__U, __A, __B);
}

__m128i test_mm_maskz_and_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_and_epi64
  //CHECK: @llvm.x86.avx512.mask.pand.q.128
  return _mm_maskz_and_epi64(__U, __A, __B);
}

__m256i test_mm256_mask_andnot_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
        __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_andnot_epi64
  //CHECK: @llvm.x86.avx512.mask.pandn.q.256
  return _mm256_mask_andnot_epi64(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_andnot_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_andnot_epi64
  //CHECK: @llvm.x86.avx512.mask.pandn.q.256
  return _mm256_maskz_andnot_epi64(__U, __A, __B);
}

__m128i test_mm_mask_andnot_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
           __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_andnot_epi64
  //CHECK: @llvm.x86.avx512.mask.pandn.q.128
  return _mm_mask_andnot_epi64(__W,__U, __A, __B);
}

__m128i test_mm_maskz_andnot_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_andnot_epi64
  //CHECK: @llvm.x86.avx512.mask.pandn.q.128
  return _mm_maskz_andnot_epi64(__U, __A, __B);
}

__m256i test_mm256_mask_or_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
          __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_or_epi64
  //CHECK: @llvm.x86.avx512.mask.por.q.256
  return _mm256_mask_or_epi64(__W,__U, __A, __B);
}

__m256i test_mm256_maskz_or_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_or_epi64
  //CHECK: @llvm.x86.avx512.mask.por.q.256
  return _mm256_maskz_or_epi64(__U, __A, __B);
}

__m128i test_mm_mask_or_epi64 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_or_epi64
  //CHECK: @llvm.x86.avx512.mask.por.q.128
  return _mm_mask_or_epi64(__W, __U, __A, __B);
}

__m128i test_mm_maskz_or_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
//CHECK-LABEL: @test_mm_maskz_or_epi64
  //CHECK: @llvm.x86.avx512.mask.por.q.128
  return _mm_maskz_or_epi64( __U, __A, __B);
}

__m256i test_mm256_mask_xor_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
          __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_xor_epi64
  //CHECK: @llvm.x86.avx512.mask.pxor.q.256
  return _mm256_mask_xor_epi64(__W,__U, __A, __B);
}

__m256i test_mm256_maskz_xor_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_xor_epi64
  //CHECK: @llvm.x86.avx512.mask.pxor.q.256
  return _mm256_maskz_xor_epi64(__U, __A, __B);
}

__m128i test_mm_mask_xor_epi64 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_xor_epi64
  //CHECK: @llvm.x86.avx512.mask.pxor.q.128
  return _mm_mask_xor_epi64(__W, __U, __A, __B);
}

__m128i test_mm_maskz_xor_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_xor_epi64
  //CHECK: @llvm.x86.avx512.mask.pxor.q.128
  return _mm_maskz_xor_epi64( __U, __A, __B);
}

__mmask8 test_mm256_cmp_ps_mask(__m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_cmp_ps_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.ps.256
  return (__mmask8)_mm256_cmp_ps_mask(__A, __B, 0);
}

__mmask8 test_mm256_mask_cmp_ps_mask(__mmask8 m, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_cmp_ps_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.ps.256
  return _mm256_mask_cmp_ps_mask(m, __A, __B, 0);
}

__mmask8 test_mm128_cmp_ps_mask(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm128_cmp_ps_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.ps.128
  return (__mmask8)_mm128_cmp_ps_mask(__A, __B, 0);
}

__mmask8 test_mm128_mask_cmp_ps_mask(__mmask8 m, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm128_mask_cmp_ps_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.ps.128
  return _mm128_mask_cmp_ps_mask(m, __A, __B, 0);
}

__mmask8 test_mm256_cmp_pd_mask(__m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_cmp_pd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.pd.256
  return (__mmask8)_mm256_cmp_pd_mask(__A, __B, 0);
}

__mmask8 test_mm256_mask_cmp_pd_mask(__mmask8 m, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_cmp_pd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.pd.256
  return _mm256_mask_cmp_pd_mask(m, __A, __B, 0);
}

__mmask8 test_mm128_cmp_pd_mask(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm128_cmp_pd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.pd.128
  return (__mmask8)_mm128_cmp_pd_mask(__A, __B, 0);
}

__mmask8 test_mm128_mask_cmp_pd_mask(__mmask8 m, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm128_mask_cmp_pd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.pd.128
  return _mm128_mask_cmp_pd_mask(m, __A, __B, 0);
}
