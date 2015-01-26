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
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 0, i8 -1)
  return (__mmask8)_mm_cmpeq_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 0, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpeq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 0, i8 -1)
  return (__mmask8)_mm_cmpeq_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 0, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpeq_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 5, i8 -1)
  return (__mmask8)_mm_cmpge_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 5, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpge_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 5, i8 -1)
  return (__mmask8)_mm_cmpge_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 5, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpge_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 5, i8 -1)
  return (__mmask8)_mm256_cmpge_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 5, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpge_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 5, i8 -1)
  return (__mmask8)_mm256_cmpge_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 5, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpge_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 5, i8 -1)
  return (__mmask8)_mm_cmpge_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 5, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpge_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 5, i8 -1)
  return (__mmask8)_mm_cmpge_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 5, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpge_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 5, i8 -1)
  return (__mmask8)_mm256_cmpge_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 5, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpge_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 5, i8 -1)
  return (__mmask8)_mm256_cmpge_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 5, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpge_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 6, i8 -1)
  return (__mmask8)_mm_cmpgt_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 6, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpgt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 6, i8 -1)
  return (__mmask8)_mm_cmpgt_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 6, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpgt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpgt_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 6, i8 -1)
  return (__mmask8)_mm256_cmpgt_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 6, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpgt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpgt_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 6, i8 -1)
  return (__mmask8)_mm256_cmpgt_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 6, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpgt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 2, i8 -1)
  return (__mmask8)_mm_cmple_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 2, i8 {{.*}})
  return (__mmask8)_mm_mask_cmple_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 2, i8 -1)
  return (__mmask8)_mm_cmple_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 2, i8 {{.*}})
  return (__mmask8)_mm_mask_cmple_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 2, i8 -1)
  return (__mmask8)_mm256_cmple_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 2, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmple_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 2, i8 -1)
  return (__mmask8)_mm256_cmple_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 2, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmple_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 2, i8 -1)
  return (__mmask8)_mm_cmple_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 2, i8 {{.*}})
  return (__mmask8)_mm_mask_cmple_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 2, i8 -1)
  return (__mmask8)_mm_cmple_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 2, i8 {{.*}})
  return (__mmask8)_mm_mask_cmple_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 2, i8 -1)
  return (__mmask8)_mm256_cmple_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 2, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmple_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 2, i8 -1)
  return (__mmask8)_mm256_cmple_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 2, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmple_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 1, i8 -1)
  return (__mmask8)_mm_cmplt_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 1, i8 {{.*}})
  return (__mmask8)_mm_mask_cmplt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 1, i8 -1)
  return (__mmask8)_mm_cmplt_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 1, i8 {{.*}})
  return (__mmask8)_mm_mask_cmplt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 1, i8 -1)
  return (__mmask8)_mm256_cmplt_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 1, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmplt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 1, i8 -1)
  return (__mmask8)_mm256_cmplt_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 1, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmplt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 1, i8 -1)
  return (__mmask8)_mm_cmplt_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 1, i8 {{.*}})
  return (__mmask8)_mm_mask_cmplt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 1, i8 -1)
  return (__mmask8)_mm_cmplt_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 1, i8 {{.*}})
  return (__mmask8)_mm_mask_cmplt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 1, i8 -1)
  return (__mmask8)_mm256_cmplt_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 1, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmplt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 1, i8 -1)
  return (__mmask8)_mm256_cmplt_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 1, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmplt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 4, i8 -1)
  return (__mmask8)_mm_cmpneq_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 4, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpneq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 4, i8 -1)
  return (__mmask8)_mm_cmpneq_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 4, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpneq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 4, i8 -1)
  return (__mmask8)_mm256_cmpneq_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 4, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpneq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 4, i8 -1)
  return (__mmask8)_mm256_cmpneq_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 4, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpneq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 4, i8 -1)
  return (__mmask8)_mm_cmpneq_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 4, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpneq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 4, i8 -1)
  return (__mmask8)_mm_cmpneq_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 4, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpneq_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 4, i8 -1)
  return (__mmask8)_mm256_cmpneq_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 4, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpneq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 4, i8 -1)
  return (__mmask8)_mm256_cmpneq_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 4, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpneq_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmp_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 7, i8 -1)
  return (__mmask8)_mm_cmp_epi32_mask(__a, __b, 7);
}

__mmask8 test_mm_mask_cmp_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 7, i8 {{.*}})
  return (__mmask8)_mm_mask_cmp_epi32_mask(__u, __a, __b, 7);
}

__mmask8 test_mm_cmp_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 7, i8 -1)
  return (__mmask8)_mm_cmp_epi64_mask(__a, __b, 7);
}

__mmask8 test_mm_mask_cmp_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 7, i8 {{.*}})
  return (__mmask8)_mm_mask_cmp_epi64_mask(__u, __a, __b, 7);
}

__mmask8 test_mm256_cmp_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 7, i8 -1)
  return (__mmask8)_mm256_cmp_epi32_mask(__a, __b, 7);
}

__mmask8 test_mm256_mask_cmp_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 7, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmp_epi32_mask(__u, __a, __b, 7);
}

__mmask8 test_mm256_cmp_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 7, i8 -1)
  return (__mmask8)_mm256_cmp_epi64_mask(__a, __b, 7);
}

__mmask8 test_mm256_mask_cmp_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 7, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmp_epi64_mask(__u, __a, __b, 7);
}

__mmask8 test_mm_cmp_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 7, i8 -1)
  return (__mmask8)_mm_cmp_epu32_mask(__a, __b, 7);
}

__mmask8 test_mm_mask_cmp_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i8 7, i8 {{.*}})
  return (__mmask8)_mm_mask_cmp_epu32_mask(__u, __a, __b, 7);
}

__mmask8 test_mm_cmp_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 7, i8 -1)
  return (__mmask8)_mm_cmp_epu64_mask(__a, __b, 7);
}

__mmask8 test_mm_mask_cmp_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i8 7, i8 {{.*}})
  return (__mmask8)_mm_mask_cmp_epu64_mask(__u, __a, __b, 7);
}

__mmask8 test_mm256_cmp_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 7, i8 -1)
  return (__mmask8)_mm256_cmp_epu32_mask(__a, __b, 7);
}

__mmask8 test_mm256_mask_cmp_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i8 7, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmp_epu32_mask(__u, __a, __b, 7);
}

__mmask8 test_mm256_cmp_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 7, i8 -1)
  return (__mmask8)_mm256_cmp_epu64_mask(__a, __b, 7);
}

__mmask8 test_mm256_mask_cmp_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i8 7, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmp_epu64_mask(__u, __a, __b, 7);
}
