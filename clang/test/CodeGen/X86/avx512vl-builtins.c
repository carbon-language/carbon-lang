// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -fexperimental-new-pass-manager -triple=x86_64-apple-darwin -target-feature +avx512f -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s

#include <immintrin.h>

__mmask8 test_mm_cmpeq_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epu32_mask
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return (__mmask8)_mm_cmpeq_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epu32_mask
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpeq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epu64_mask
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: shufflevector <2 x i1> %{{.*}}, <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  return (__mmask8)_mm_cmpeq_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epu64_mask
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpeq_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epi32_mask
  // CHECK: icmp sge <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpge_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epi32_mask
  // CHECK: icmp sge <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpge_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epi64_mask
  // CHECK: icmp sge <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpge_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epi64_mask
  // CHECK: icmp sge <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpge_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epi32_mask
  // CHECK: icmp sge <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpge_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epi32_mask
  // CHECK: icmp sge <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpge_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epi64_mask
  // CHECK: icmp sge <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpge_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epi64_mask
  // CHECK: icmp sge <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpge_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epu32_mask
  // CHECK: icmp uge <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpge_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epu32_mask
  // CHECK: icmp uge <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpge_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epu64_mask
  // CHECK: icmp uge <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpge_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epu64_mask
  // CHECK: icmp uge <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpge_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epu32_mask
  // CHECK: icmp uge <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpge_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epu32_mask
  // CHECK: icmp uge <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpge_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epu64_mask
  // CHECK: icmp uge <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpge_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epu64_mask
  // CHECK: icmp uge <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpge_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epu32_mask
  // CHECK: icmp ugt <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpgt_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epu32_mask
  // CHECK: icmp ugt <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpgt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epu64_mask
  // CHECK: icmp ugt <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpgt_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epu64_mask
  // CHECK: icmp ugt <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpgt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpgt_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epu32_mask
  // CHECK: icmp ugt <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpgt_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epu32_mask
  // CHECK: icmp ugt <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpgt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpgt_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epu64_mask
  // CHECK: icmp ugt <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpgt_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epu64_mask
  // CHECK: icmp ugt <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpgt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epi32_mask
  // CHECK: icmp sle <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmple_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epi32_mask
  // CHECK: icmp sle <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmple_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epi64_mask
  // CHECK: icmp sle <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmple_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epi64_mask
  // CHECK: icmp sle <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmple_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epi32_mask
  // CHECK: icmp sle <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmple_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epi32_mask
  // CHECK: icmp sle <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmple_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epi64_mask
  // CHECK: icmp sle <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmple_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epi64_mask
  // CHECK: icmp sle <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmple_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epu32_mask
  // CHECK: icmp ule <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmple_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epu32_mask
  // CHECK: icmp ule <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmple_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epu64_mask
  // CHECK: icmp ule <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmple_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epu64_mask
  // CHECK: icmp ule <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmple_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epu32_mask
  // CHECK: icmp ule <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmple_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epu32_mask
  // CHECK: icmp ule <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmple_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epu64_mask
  // CHECK: icmp ule <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmple_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epu64_mask
  // CHECK: icmp ule <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmple_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epi32_mask
  // CHECK: icmp slt <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmplt_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epi32_mask
  // CHECK: icmp slt <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmplt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epi64_mask
  // CHECK: icmp slt <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmplt_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epi64_mask
  // CHECK: icmp slt <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmplt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epi32_mask
  // CHECK: icmp slt <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmplt_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epi32_mask
  // CHECK: icmp slt <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmplt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epi64_mask
  // CHECK: icmp slt <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmplt_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epi64_mask
  // CHECK: icmp slt <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmplt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epu32_mask
  // CHECK: icmp ult <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmplt_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epu32_mask
  // CHECK: icmp ult <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmplt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epu64_mask
  // CHECK: icmp ult <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmplt_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epu64_mask
  // CHECK: icmp ult <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmplt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epu32_mask
  // CHECK: icmp ult <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmplt_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epu32_mask
  // CHECK: icmp ult <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmplt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epu64_mask
  // CHECK: icmp ult <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmplt_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epu64_mask
  // CHECK: icmp ult <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmplt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epi32_mask
  // CHECK: icmp ne <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpneq_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epi32_mask
  // CHECK: icmp ne <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpneq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epi64_mask
  // CHECK: icmp ne <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpneq_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epi64_mask
  // CHECK: icmp ne <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpneq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epi32_mask
  // CHECK: icmp ne <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpneq_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epi32_mask
  // CHECK: icmp ne <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpneq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epi64_mask
  // CHECK: icmp ne <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpneq_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epi64_mask
  // CHECK: icmp ne <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpneq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epu32_mask
  // CHECK: icmp ne <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpneq_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epu32_mask
  // CHECK: icmp ne <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpneq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epu64_mask
  // CHECK: icmp ne <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpneq_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epu64_mask
  // CHECK: icmp ne <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpneq_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epu32_mask
  // CHECK: icmp ne <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpneq_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epu32_mask
  // CHECK: icmp ne <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpneq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epu64_mask
  // CHECK: icmp ne <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpneq_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epu64_mask
  // CHECK: icmp ne <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpneq_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmp_eq_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_eq_epi32_mask
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmp_epi32_mask(__a, __b, _MM_CMPINT_EQ);
}

__mmask8 test_mm_mask_cmp_lt_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_lt_epi32_mask
  // CHECK: icmp slt <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmp_epi32_mask(__u, __a, __b, _MM_CMPINT_LT);
}

__mmask8 test_mm_cmp_lt_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_lt_epi64_mask
  // CHECK: icmp slt <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmp_epi64_mask(__a, __b, _MM_CMPINT_LT);
}

__mmask8 test_mm_mask_cmp_eq_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_eq_epi64_mask
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmp_epi64_mask(__u, __a, __b, _MM_CMPINT_EQ);
}

__mmask8 test_mm256_cmp_eq_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_eq_epi32_mask
  // CHECK: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmp_epi32_mask(__a, __b, _MM_CMPINT_EQ);
}

__mmask8 test_mm256_mask_cmp_le_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_le_epi32_mask
  // CHECK: icmp sle <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmp_epi32_mask(__u, __a, __b, _MM_CMPINT_LE);
}

__mmask8 test_mm256_cmp_eq_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_eq_epi64_mask
  // CHECK: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmp_epi64_mask(__a, __b, _MM_CMPINT_EQ);
}

__mmask8 test_mm256_mask_cmp_eq_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_eq_epi64_mask
  // CHECK: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmp_epi64_mask(__u, __a, __b, _MM_CMPINT_EQ);
}

__mmask8 test_mm_cmp_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epu32_mask
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmp_epu32_mask(__a, __b, 0);
}

__mmask8 test_mm_mask_cmp_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epu32_mask
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmp_epu32_mask(__u, __a, __b, 0);
}

__mmask8 test_mm_cmp_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epu64_mask
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmp_epu64_mask(__a, __b, 0);
}

__mmask8 test_mm_mask_cmp_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epu64_mask
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmp_epu64_mask(__u, __a, __b, 0);
}

__mmask8 test_mm256_cmp_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epu32_mask
  // CHECK: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmp_epu32_mask(__a, __b, 0);
}

__mmask8 test_mm256_mask_cmp_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epu32_mask
  // CHECK: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmp_epu32_mask(__u, __a, __b, 0);
}

__mmask8 test_mm256_cmp_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epu64_mask
  // CHECK: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmp_epu64_mask(__a, __b, 0);
}

__mmask8 test_mm256_mask_cmp_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epu64_mask
  // CHECK: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmp_epu64_mask(__u, __a, __b, 0);
}

__m256i test_mm256_mask_add_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_add_epi32
  //CHECK: add <8 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_add_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_add_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_add_epi32
  //CHECK: add <8 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_add_epi32(__U, __A, __B);
}

__m256i test_mm256_mask_add_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_add_epi64
  //CHECK: add <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_add_epi64(__W,__U,__A,__B);
}

__m256i test_mm256_maskz_add_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_add_epi64
  //CHECK: add <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_add_epi64 (__U,__A,__B);
}

__m256i test_mm256_mask_sub_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_sub_epi32
  //CHECK: sub <8 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_sub_epi32 (__W,__U,__A,__B);
}

__m256i test_mm256_maskz_sub_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_sub_epi32
  //CHECK: sub <8 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_sub_epi32 (__U,__A,__B);
}

__m256i test_mm256_mask_sub_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_sub_epi64
  //CHECK: sub <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_sub_epi64 (__W,__U,__A,__B);
}

__m256i test_mm256_maskz_sub_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_sub_epi64
  //CHECK: sub <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_sub_epi64 (__U,__A,__B);
}

__m128i test_mm_mask_add_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_add_epi32
  //CHECK: add <4 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_add_epi32(__W,__U,__A,__B);
}


__m128i test_mm_maskz_add_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_add_epi32
  //CHECK: add <4 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_add_epi32 (__U,__A,__B);
}

__m128i test_mm_mask_add_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_add_epi64
  //CHECK: add <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_add_epi64 (__W,__U,__A,__B);
}

__m128i test_mm_maskz_add_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_add_epi64
  //CHECK: add <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_add_epi64 (__U,__A,__B);
}

__m128i test_mm_mask_sub_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_sub_epi32
  //CHECK: sub <4 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_sub_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_sub_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_sub_epi32
  //CHECK: sub <4 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_sub_epi32(__U, __A, __B);
}

__m128i test_mm_mask_sub_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_sub_epi64
  //CHECK: sub <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_sub_epi64 (__W, __U, __A, __B);
}

__m128i test_mm_maskz_sub_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_sub_epi64
  //CHECK: sub <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_sub_epi64 (__U, __A, __B);
}

__m256i test_mm256_mask_mul_epi32 (__m256i __W, __mmask8 __M, __m256i __X,
           __m256i __Y) {
  //CHECK-LABEL: @test_mm256_mask_mul_epi32
  //CHECK: shl <4 x i64> %{{.*}}, <i64 32, i64 32, i64 32, i64 32>
  //CHECK: ashr <4 x i64> %{{.*}}, <i64 32, i64 32, i64 32, i64 32>
  //CHECK: shl <4 x i64> %{{.*}}, <i64 32, i64 32, i64 32, i64 32>
  //CHECK: ashr <4 x i64> %{{.*}}, <i64 32, i64 32, i64 32, i64 32>
  //CHECK: mul <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_mul_epi32(__W, __M, __X, __Y);
}

__m256i test_mm256_maskz_mul_epi32 (__mmask8 __M, __m256i __X, __m256i __Y) {
  //CHECK-LABEL: @test_mm256_maskz_mul_epi32
  //CHECK: shl <4 x i64> %{{.*}}, <i64 32, i64 32, i64 32, i64 32>
  //CHECK: ashr <4 x i64> %{{.*}}, <i64 32, i64 32, i64 32, i64 32>
  //CHECK: shl <4 x i64> %{{.*}}, <i64 32, i64 32, i64 32, i64 32>
  //CHECK: ashr <4 x i64> %{{.*}}, <i64 32, i64 32, i64 32, i64 32>
  //CHECK: mul <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_mul_epi32(__M, __X, __Y);
}


__m128i test_mm_mask_mul_epi32 (__m128i __W, __mmask8 __M, __m128i __X,
        __m128i __Y) {
  //CHECK-LABEL: @test_mm_mask_mul_epi32
  //CHECK: shl <2 x i64> %{{.*}}, <i64 32, i64 32>
  //CHECK: ashr <2 x i64> %{{.*}}, <i64 32, i64 32>
  //CHECK: shl <2 x i64> %{{.*}}, <i64 32, i64 32>
  //CHECK: ashr <2 x i64> %{{.*}}, <i64 32, i64 32>
  //CHECK: mul <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_mul_epi32(__W, __M, __X, __Y);
}

__m128i test_mm_maskz_mul_epi32 (__mmask8 __M, __m128i __X, __m128i __Y) {
  //CHECK-LABEL: @test_mm_maskz_mul_epi32
  //CHECK: shl <2 x i64> %{{.*}}, <i64 32, i64 32>
  //CHECK: ashr <2 x i64> %{{.*}}, <i64 32, i64 32>
  //CHECK: shl <2 x i64> %{{.*}}, <i64 32, i64 32>
  //CHECK: ashr <2 x i64> %{{.*}}, <i64 32, i64 32>
  //CHECK: mul <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_mul_epi32(__M, __X, __Y);
}

__m256i test_mm256_mask_mul_epu32 (__m256i __W, __mmask8 __M, __m256i __X,
           __m256i __Y) {
  //CHECK-LABEL: @test_mm256_mask_mul_epu32
  //CHECK: and <4 x i64> %{{.*}}, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>
  //CHECK: and <4 x i64> %{{.*}}, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>
  //CHECK: mul <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_mul_epu32(__W, __M, __X, __Y);
}

__m256i test_mm256_maskz_mul_epu32 (__mmask8 __M, __m256i __X, __m256i __Y) {
  //CHECK-LABEL: @test_mm256_maskz_mul_epu32
  //CHECK: and <4 x i64> %{{.*}}, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>
  //CHECK: and <4 x i64> %{{.*}}, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>
  //CHECK: mul <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_mul_epu32(__M, __X, __Y);
}

__m128i test_mm_mask_mul_epu32 (__m128i __W, __mmask8 __M, __m128i __X,
        __m128i __Y) {
  //CHECK-LABEL: @test_mm_mask_mul_epu32
  //CHECK: and <2 x i64> %{{.*}}, <i64 4294967295, i64 4294967295>
  //CHECK: and <2 x i64> %{{.*}}, <i64 4294967295, i64 4294967295>
  //CHECK: mul <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_mul_epu32(__W, __M, __X, __Y);
}

__m128i test_mm_maskz_mul_epu32 (__mmask8 __M, __m128i __X, __m128i __Y) {
  //CHECK-LABEL: @test_mm_maskz_mul_epu32
  //CHECK: and <2 x i64> %{{.*}}, <i64 4294967295, i64 4294967295>
  //CHECK: and <2 x i64> %{{.*}}, <i64 4294967295, i64 4294967295>
  //CHECK: mul <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_mul_epu32(__M, __X, __Y);
}

__m128i test_mm_maskz_mullo_epi32 (__mmask8 __M, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_mullo_epi32
  //CHECK: mul <4 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_mullo_epi32(__M, __A, __B);
}

__m128i test_mm_mask_mullo_epi32 (__m128i __W, __mmask8 __M, __m128i __A,
          __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_mullo_epi32
  //CHECK: mul <4 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_mullo_epi32(__W, __M, __A, __B);
}

__m256i test_mm256_maskz_mullo_epi32 (__mmask8 __M, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_mullo_epi32
  //CHECK: mul <8 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_mullo_epi32(__M, __A, __B);
}

__m256i test_mm256_mask_mullo_epi32 (__m256i __W, __mmask8 __M, __m256i __A,
       __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_mullo_epi32
  //CHECK: mul <8 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_mullo_epi32(__W, __M, __A, __B);
}

__m256i test_mm256_and_epi32 (__m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_and_epi32
  //CHECK: and <8 x i32> %{{.*}}, %{{.*}}
  return _mm256_and_epi32(__A, __B);
}

__m256i test_mm256_mask_and_epi32 (__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_and_epi32
  //CHECK: and <8 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_and_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_and_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_and_epi32
  //CHECK: and <8 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_and_epi32(__U, __A, __B);
}

__m128i test_mm_and_epi32 (__m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_and_epi32
  //CHECK: and <4 x i32> %{{.*}}, %{{.*}}
  return _mm_and_epi32(__A, __B);
}

__m128i test_mm_mask_and_epi32 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_and_epi32
  //CHECK: and <4 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_and_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_and_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_and_epi32
  //CHECK: and <4 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_and_epi32(__U, __A, __B);
}

__m256i test_mm256_andnot_epi32 (__m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_andnot_epi32
  //CHECK: xor <8 x i32> %{{.*}}, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  //CHECK: and <8 x i32> %{{.*}}, %{{.*}}
  return _mm256_andnot_epi32(__A, __B);
}

__m256i test_mm256_mask_andnot_epi32 (__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_andnot_epi32
  //CHECK: xor <8 x i32> %{{.*}}, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  //CHECK: and <8 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_andnot_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_andnot_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_andnot_epi32
  //CHECK: xor <8 x i32> %{{.*}}, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  //CHECK: and <8 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_andnot_epi32(__U, __A, __B);
}

__m128i test_mm_andnot_epi32 (__m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_andnot_epi32
  //CHECK: xor <4 x i32> %{{.*}}, <i32 -1, i32 -1, i32 -1, i32 -1>
  //CHECK: and <4 x i32> %{{.*}}, %{{.*}}
  return _mm_andnot_epi32(__A, __B);
}

__m128i test_mm_mask_andnot_epi32 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_andnot_epi32
  //CHECK: xor <4 x i32> %{{.*}}, <i32 -1, i32 -1, i32 -1, i32 -1>
  //CHECK: and <4 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_andnot_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_andnot_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_andnot_epi32
  //CHECK: xor <4 x i32> %{{.*}}, <i32 -1, i32 -1, i32 -1, i32 -1>
  //CHECK: and <4 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_andnot_epi32(__U, __A, __B);
}

__m256i test_mm256_or_epi32 (__m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_or_epi32
  //CHECK: or <8 x i32> %{{.*}}, %{{.*}}
  return _mm256_or_epi32(__A, __B);
}

__m256i test_mm256_mask_or_epi32 (__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_or_epi32
  //CHECK: or <8 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_or_epi32(__W, __U, __A, __B);
}

 __m256i test_mm256_maskz_or_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_or_epi32
  //CHECK: or <8 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_or_epi32(__U, __A, __B);
}

__m128i test_mm_or_epi32 (__m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_or_epi32
  //CHECK: or <4 x i32> %{{.*}}, %{{.*}}
  return _mm_or_epi32(__A, __B);
}

__m128i test_mm_mask_or_epi32 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_or_epi32
  //CHECK: or <4 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_or_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_or_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_or_epi32
  //CHECK: or <4 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_or_epi32(__U, __A, __B);
}

__m256i test_mm256_xor_epi32 (__m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_xor_epi32
  //CHECK: or <8 x i32> %{{.*}}, %{{.*}}
  return _mm256_xor_epi32(__A, __B);
}

__m256i test_mm256_mask_xor_epi32 (__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_xor_epi32
  //CHECK: xor <8 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_xor_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_xor_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_xor_epi32
  //CHECK: xor <8 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_xor_epi32(__U, __A, __B);
}

__m128i test_mm_xor_epi32 (__m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_xor_epi32
  //CHECK: xor <4 x i32> %{{.*}}, %{{.*}}
  return _mm_xor_epi32(__A, __B);
}

__m128i test_mm_mask_xor_epi32 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_xor_epi32
  //CHECK: xor <4 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_xor_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_xor_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_xor_epi32
  //CHECK: xor <4 x i32> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_xor_epi32(__U, __A, __B);
}

__m256i test_mm256_and_epi64 (__m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_and_epi64
  //CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  return _mm256_and_epi64(__A, __B);
}

__m256i test_mm256_mask_and_epi64 (__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_and_epi64
  //CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_and_epi64(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_and_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_and_epi64
  //CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_and_epi64(__U, __A, __B);
}

__m128i test_mm_and_epi64 (__m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_and_epi64
  //CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  return _mm_and_epi64(__A, __B);
}

__m128i test_mm_mask_and_epi64 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_and_epi64
  //CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_and_epi64(__W,__U, __A, __B);
}

__m128i test_mm_maskz_and_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_and_epi64
  //CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_and_epi64(__U, __A, __B);
}

__m256i test_mm256_andnot_epi64 (__m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_andnot_epi64
  //CHECK: xor <4 x i64> %{{.*}}, <i64 -1, i64 -1, i64 -1, i64 -1>
  //CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  return _mm256_andnot_epi64(__A, __B);
}

__m256i test_mm256_mask_andnot_epi64 (__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_andnot_epi64
  //CHECK: xor <4 x i64> %{{.*}}, <i64 -1, i64 -1, i64 -1, i64 -1>
  //CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_andnot_epi64(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_andnot_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_andnot_epi64
  //CHECK: xor <4 x i64> %{{.*}}, <i64 -1, i64 -1, i64 -1, i64 -1>
  //CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_andnot_epi64(__U, __A, __B);
}

__m128i test_mm_andnot_epi64 (__m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_andnot_epi64
  //CHECK: xor <2 x i64> %{{.*}}, <i64 -1, i64 -1>
  //CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  return _mm_andnot_epi64(__A, __B);
}

__m128i test_mm_mask_andnot_epi64 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_andnot_epi64
  //CHECK: xor <2 x i64> %{{.*}}, <i64 -1, i64 -1>
  //CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_andnot_epi64(__W,__U, __A, __B);
}

__m128i test_mm_maskz_andnot_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_andnot_epi64
  //CHECK: xor <2 x i64> %{{.*}}, <i64 -1, i64 -1>
  //CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_andnot_epi64(__U, __A, __B);
}

__m256i test_mm256_or_epi64 (__m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_or_epi64
  //CHECK: or <4 x i64> %{{.*}}, %{{.*}}
  return _mm256_or_epi64(__A, __B);
}

__m256i test_mm256_mask_or_epi64 (__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_or_epi64
  //CHECK: or <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_or_epi64(__W,__U, __A, __B);
}

__m256i test_mm256_maskz_or_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_or_epi64
  //CHECK: or <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_or_epi64(__U, __A, __B);
}

__m128i test_mm_or_epi64 (__m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_or_epi64
  //CHECK: or <2 x i64> %{{.*}}, %{{.*}}
  return _mm_or_epi64(__A, __B);
}

__m128i test_mm_mask_or_epi64 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_or_epi64
  //CHECK: or <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_or_epi64(__W, __U, __A, __B);
}

__m128i test_mm_maskz_or_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_or_epi64
  //CHECK: or <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_or_epi64( __U, __A, __B);
}

__m256i test_mm256_xor_epi64 (__m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_xor_epi64
  //CHECK: xor <4 x i64> %{{.*}}, %{{.*}}
  return _mm256_xor_epi64(__A, __B);
}

__m256i test_mm256_mask_xor_epi64 (__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_xor_epi64
  //CHECK: xor <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_xor_epi64(__W,__U, __A, __B);
}

__m256i test_mm256_maskz_xor_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_xor_epi64
  //CHECK: xor <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_xor_epi64(__U, __A, __B);
}

__m128i test_mm_xor_epi64 (__m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_xor_epi64
  //CHECK: xor <2 x i64> %{{.*}}, %{{.*}}
  return _mm_xor_epi64(__A, __B);
}

__m128i test_mm_mask_xor_epi64 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_xor_epi64
  //CHECK: xor <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_xor_epi64(__W, __U, __A, __B);
}

__m128i test_mm_maskz_xor_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_xor_epi64
  //CHECK: xor <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_xor_epi64( __U, __A, __B);
}

__mmask8 test_mm256_cmp_ps_mask_eq_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: @test_mm256_cmp_ps_mask_eq_oq
  // CHECK: fcmp oeq <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_lt_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_lt_os
  // CHECK: fcmp olt <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm256_cmp_ps_mask_le_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_le_os
  // CHECK: fcmp ole <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_LE_OS);
}

__mmask8 test_mm256_cmp_ps_mask_unord_q(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_unord_q
  // CHECK: fcmp uno <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm256_cmp_ps_mask_neq_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_neq_uq
  // CHECK: fcmp une <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_nlt_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nlt_us
  // CHECK: fcmp uge <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_NLT_US);
}

__mmask8 test_mm256_cmp_ps_mask_nle_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nle_us
  // CHECK: fcmp ugt <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_NLE_US);
}

__mmask8 test_mm256_cmp_ps_mask_ord_q(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ord_q
  // CHECK: fcmp ord <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_ORD_Q);
}

__mmask8 test_mm256_cmp_ps_mask_eq_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_eq_uq
  // CHECK: fcmp ueq <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_nge_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nge_us
  // CHECK: fcmp ult <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_NGE_US);
}

__mmask8 test_mm256_cmp_ps_mask_ngt_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ngt_us
  // CHECK: fcmp ule <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_NGT_US);
}

__mmask8 test_mm256_cmp_ps_mask_false_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_false_oq
  // CHECK: fcmp false <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_neq_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_neq_oq
  // CHECK: fcmp one <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_ge_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ge_os
  // CHECK: fcmp oge <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_GE_OS);
}

__mmask8 test_mm256_cmp_ps_mask_gt_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_gt_os
  // CHECK: fcmp ogt <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_GT_OS);
}

__mmask8 test_mm256_cmp_ps_mask_true_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_true_uq
  // CHECK: fcmp true <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_eq_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_eq_os
  // CHECK: fcmp oeq <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_EQ_OS);
}

__mmask8 test_mm256_cmp_ps_mask_lt_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_lt_oq
  // CHECK: fcmp olt <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_LT_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_le_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_le_oq
  // CHECK: fcmp ole <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_LE_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_unord_s(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_unord_s
  // CHECK: fcmp uno <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_UNORD_S);
}

__mmask8 test_mm256_cmp_ps_mask_neq_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_neq_us
  // CHECK: fcmp une <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_NEQ_US);
}

__mmask8 test_mm256_cmp_ps_mask_nlt_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nlt_uq
  // CHECK: fcmp uge <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_nle_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nle_uq
  // CHECK: fcmp ugt <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_ord_s(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ord_s
  // CHECK: fcmp ord <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_ORD_S);
}

__mmask8 test_mm256_cmp_ps_mask_eq_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_eq_us
  // CHECK: fcmp ueq <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_EQ_US);
}

__mmask8 test_mm256_cmp_ps_mask_nge_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_nge_uq
  // CHECK: fcmp ult <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_ngt_uq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ngt_uq
  // CHECK: fcmp ule <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm256_cmp_ps_mask_false_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_false_os
  // CHECK: fcmp false <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm256_cmp_ps_mask_neq_os(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_neq_os
  // CHECK: fcmp one <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm256_cmp_ps_mask_ge_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_ge_oq
  // CHECK: fcmp oge <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_GE_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_gt_oq(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_gt_oq
  // CHECK: fcmp ogt <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_GT_OQ);
}

__mmask8 test_mm256_cmp_ps_mask_true_us(__m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_cmp_ps_mask_true_us
  // CHECK: fcmp true <8 x float> %{{.*}}, %{{.*}}
  return _mm256_cmp_ps_mask(a, b, _CMP_TRUE_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_eq_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_ps_mask_eq_oq
  // CHECK: [[CMP:%.*]] = fcmp oeq <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_lt_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_lt_os
  // CHECK: [[CMP:%.*]] = fcmp olt <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_LT_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_le_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_le_os
  // CHECK: [[CMP:%.*]] = fcmp ole <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_LE_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_unord_q(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_unord_q
  // CHECK: [[CMP:%.*]] = fcmp uno <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm256_mask_cmp_ps_mask_neq_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_neq_uq
  // CHECK: [[CMP:%.*]] = fcmp une <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nlt_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nlt_us
  // CHECK: [[CMP:%.*]] = fcmp uge <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NLT_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nle_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nle_us
  // CHECK: [[CMP:%.*]] = fcmp ugt <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NLE_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ord_q(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ord_q
  // CHECK: [[CMP:%.*]] = fcmp ord <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_ORD_Q);
}

__mmask8 test_mm256_mask_cmp_ps_mask_eq_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_eq_uq
  // CHECK: [[CMP:%.*]] = fcmp ueq <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nge_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nge_us
  // CHECK: [[CMP:%.*]] = fcmp ult <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NGE_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ngt_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ngt_us
  // CHECK: [[CMP:%.*]] = fcmp ule <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NGT_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_false_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_false_oq
  // CHECK: [[CMP:%.*]] = fcmp false <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_neq_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_neq_oq
  // CHECK: [[CMP:%.*]] = fcmp one <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ge_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ge_os
  // CHECK: [[CMP:%.*]] = fcmp oge <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_GE_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_gt_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_gt_os
  // CHECK: [[CMP:%.*]] = fcmp ogt <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_GT_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_true_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_true_uq
  // CHECK: [[CMP:%.*]] = fcmp true <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_eq_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_eq_os
  // CHECK: [[CMP:%.*]] = fcmp oeq <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_EQ_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_lt_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_lt_oq
  // CHECK: [[CMP:%.*]] = fcmp olt <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_LT_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_le_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_le_oq
  // CHECK: [[CMP:%.*]] = fcmp ole <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_LE_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_unord_s(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_unord_s
  // CHECK: [[CMP:%.*]] = fcmp uno <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_UNORD_S);
}

__mmask8 test_mm256_mask_cmp_ps_mask_neq_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_neq_us
  // CHECK: [[CMP:%.*]] = fcmp une <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nlt_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nlt_uq
  // CHECK: [[CMP:%.*]] = fcmp uge <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nle_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nle_uq
  // CHECK: [[CMP:%.*]] = fcmp ugt <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ord_s(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ord_s
  // CHECK: [[CMP:%.*]] = fcmp ord <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_ORD_S);
}

__mmask8 test_mm256_mask_cmp_ps_mask_eq_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_eq_us
  // CHECK: [[CMP:%.*]] = fcmp ueq <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_EQ_US);
}

__mmask8 test_mm256_mask_cmp_ps_mask_nge_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_nge_uq
  // CHECK: [[CMP:%.*]] = fcmp ult <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ngt_uq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ngt_uq
  // CHECK: [[CMP:%.*]] = fcmp ule <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_false_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_false_os
  // CHECK: [[CMP:%.*]] = fcmp false <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_neq_os(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_neq_os
  // CHECK: [[CMP:%.*]] = fcmp one <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm256_mask_cmp_ps_mask_ge_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_ge_oq
  // CHECK: [[CMP:%.*]] = fcmp oge <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_GE_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_gt_oq(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_gt_oq
  // CHECK: [[CMP:%.*]] = fcmp ogt <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_GT_OQ);
}

__mmask8 test_mm256_mask_cmp_ps_mask_true_us(__mmask8 m, __m256 a, __m256 b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ps_mask_true_us
  // CHECK: [[CMP:%.*]] = fcmp true <8 x float> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ps_mask(m, a, b, _CMP_TRUE_US);
}

__mmask8 test_mm256_cmp_pd_mask_eq_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: @test_mm256_cmp_pd_mask_eq_oq
  // CHECK: fcmp oeq <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_lt_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_lt_os
  // CHECK: fcmp olt <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm256_cmp_pd_mask_le_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_le_os
  // CHECK: fcmp ole <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_LE_OS);
}

__mmask8 test_mm256_cmp_pd_mask_unord_q(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_unord_q
  // CHECK: fcmp uno <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm256_cmp_pd_mask_neq_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_neq_uq
  // CHECK: fcmp une <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_nlt_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nlt_us
  // CHECK: fcmp uge <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_NLT_US);
}

__mmask8 test_mm256_cmp_pd_mask_nle_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nle_us
  // CHECK: fcmp ugt <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_NLE_US);
}

__mmask8 test_mm256_cmp_pd_mask_ord_q(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ord_q
  // CHECK: fcmp ord <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_ORD_Q);
}

__mmask8 test_mm256_cmp_pd_mask_eq_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_eq_uq
  // CHECK: fcmp ueq <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_nge_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nge_us
  // CHECK: fcmp ult <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_NGE_US);
}

__mmask8 test_mm256_cmp_pd_mask_ngt_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ngt_us
  // CHECK: fcmp ule <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_NGT_US);
}

__mmask8 test_mm256_cmp_pd_mask_false_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_false_oq
  // CHECK: fcmp false <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_neq_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_neq_oq
  // CHECK: fcmp one <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_ge_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ge_os
  // CHECK: fcmp oge <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_GE_OS);
}

__mmask8 test_mm256_cmp_pd_mask_gt_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_gt_os
  // CHECK: fcmp ogt <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_GT_OS);
}

__mmask8 test_mm256_cmp_pd_mask_true_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_true_uq
  // CHECK: fcmp true <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_eq_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_eq_os
  // CHECK: fcmp oeq <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_EQ_OS);
}

__mmask8 test_mm256_cmp_pd_mask_lt_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_lt_oq
  // CHECK: fcmp olt <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_LT_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_le_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_le_oq
  // CHECK: fcmp ole <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_LE_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_unord_s(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_unord_s
  // CHECK: fcmp uno <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_UNORD_S);
}

__mmask8 test_mm256_cmp_pd_mask_neq_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_neq_us
  // CHECK: fcmp une <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_NEQ_US);
}

__mmask8 test_mm256_cmp_pd_mask_nlt_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nlt_uq
  // CHECK: fcmp uge <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_nle_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nle_uq
  // CHECK: fcmp ugt <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_ord_s(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ord_s
  // CHECK: fcmp ord <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_ORD_S);
}

__mmask8 test_mm256_cmp_pd_mask_eq_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_eq_us
  // CHECK: fcmp ueq <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_EQ_US);
}

__mmask8 test_mm256_cmp_pd_mask_nge_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_nge_uq
  // CHECK: fcmp ult <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_ngt_uq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ngt_uq
  // CHECK: fcmp ule <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm256_cmp_pd_mask_false_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_false_os
  // CHECK: fcmp false <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm256_cmp_pd_mask_neq_os(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_neq_os
  // CHECK: fcmp one <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm256_cmp_pd_mask_ge_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_ge_oq
  // CHECK: fcmp oge <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_GE_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_gt_oq(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_gt_oq
  // CHECK: fcmp ogt <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_GT_OQ);
}

__mmask8 test_mm256_cmp_pd_mask_true_us(__m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_cmp_pd_mask_true_us
  // CHECK: fcmp true <4 x double> %{{.*}}, %{{.*}}
  return _mm256_cmp_pd_mask(a, b, _CMP_TRUE_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_eq_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_pd_mask_eq_oq
  // CHECK: [[CMP:%.*]] = fcmp oeq <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_lt_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_lt_os
  // CHECK: [[CMP:%.*]] = fcmp olt <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_LT_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_le_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_le_os
  // CHECK: [[CMP:%.*]] = fcmp ole <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_LE_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_unord_q(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_unord_q
  // CHECK: [[CMP:%.*]] = fcmp uno <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm256_mask_cmp_pd_mask_neq_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_neq_uq
  // CHECK: [[CMP:%.*]] = fcmp une <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nlt_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nlt_us
  // CHECK: [[CMP:%.*]] = fcmp uge <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NLT_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nle_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nle_us
  // CHECK: [[CMP:%.*]] = fcmp ugt <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NLE_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ord_q(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ord_q
  // CHECK: [[CMP:%.*]] = fcmp ord <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_ORD_Q);
}

__mmask8 test_mm256_mask_cmp_pd_mask_eq_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_eq_uq
  // CHECK: [[CMP:%.*]] = fcmp ueq <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nge_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nge_us
  // CHECK: [[CMP:%.*]] = fcmp ult <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NGE_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ngt_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ngt_us
  // CHECK: [[CMP:%.*]] = fcmp ule <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NGT_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_false_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_false_oq
  // CHECK: [[CMP:%.*]] = fcmp false <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_neq_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_neq_oq
  // CHECK: [[CMP:%.*]] = fcmp one <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ge_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ge_os
  // CHECK: [[CMP:%.*]] = fcmp oge <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_GE_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_gt_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_gt_os
  // CHECK: [[CMP:%.*]] = fcmp ogt <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_GT_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_true_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_true_uq
  // CHECK: [[CMP:%.*]] = fcmp true <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_eq_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_eq_os
  // CHECK: [[CMP:%.*]] = fcmp oeq <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_EQ_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_lt_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_lt_oq
  // CHECK: [[CMP:%.*]] = fcmp olt <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_LT_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_le_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_le_oq
  // CHECK: [[CMP:%.*]] = fcmp ole <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_LE_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_unord_s(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_unord_s
  // CHECK: [[CMP:%.*]] = fcmp uno <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_UNORD_S);
}

__mmask8 test_mm256_mask_cmp_pd_mask_neq_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_neq_us
  // CHECK: [[CMP:%.*]] = fcmp une <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nlt_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nlt_uq
  // CHECK: [[CMP:%.*]] = fcmp uge <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nle_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nle_uq
  // CHECK: [[CMP:%.*]] = fcmp ugt <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ord_s(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ord_s
  // CHECK: [[CMP:%.*]] = fcmp ord <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_ORD_S);
}

__mmask8 test_mm256_mask_cmp_pd_mask_eq_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_eq_us
  // CHECK: [[CMP:%.*]] = fcmp ueq <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_EQ_US);
}

__mmask8 test_mm256_mask_cmp_pd_mask_nge_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_nge_uq
  // CHECK: [[CMP:%.*]] = fcmp ult <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ngt_uq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ngt_uq
  // CHECK: [[CMP:%.*]] = fcmp ule <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_false_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_false_os
  // CHECK: [[CMP:%.*]] = fcmp false <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_neq_os(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_neq_os
  // CHECK: [[CMP:%.*]] = fcmp one <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm256_mask_cmp_pd_mask_ge_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_ge_oq
  // CHECK: [[CMP:%.*]] = fcmp oge <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_GE_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_gt_oq(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_gt_oq
  // CHECK: [[CMP:%.*]] = fcmp ogt <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_GT_OQ);
}

__mmask8 test_mm256_mask_cmp_pd_mask_true_us(__mmask8 m, __m256d a, __m256d b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pd_mask_true_us
  // CHECK: [[CMP:%.*]] = fcmp true <4 x double> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_pd_mask(m, a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_cmp_ps_mask_eq_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: @test_mm_cmp_ps_mask_eq_oq
  // CHECK: fcmp oeq <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm_cmp_ps_mask_lt_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_lt_os
  // CHECK: fcmp olt <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm_cmp_ps_mask_le_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_le_os
  // CHECK: fcmp ole <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_LE_OS);
}

__mmask8 test_mm_cmp_ps_mask_unord_q(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_unord_q
  // CHECK: fcmp uno <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm_cmp_ps_mask_neq_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_neq_uq
  // CHECK: fcmp une <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm_cmp_ps_mask_nlt_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nlt_us
  // CHECK: fcmp uge <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_NLT_US);
}

__mmask8 test_mm_cmp_ps_mask_nle_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nle_us
  // CHECK: fcmp ugt <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_NLE_US);
}

__mmask8 test_mm_cmp_ps_mask_ord_q(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ord_q
  // CHECK: fcmp ord <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_ORD_Q);
}

__mmask8 test_mm_cmp_ps_mask_eq_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_eq_uq
  // CHECK: fcmp ueq <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm_cmp_ps_mask_nge_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nge_us
  // CHECK: fcmp ult <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_NGE_US);
}

__mmask8 test_mm_cmp_ps_mask_ngt_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ngt_us
  // CHECK: fcmp ule <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_NGT_US);
}

__mmask8 test_mm_cmp_ps_mask_false_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_false_oq
  // CHECK: fcmp false <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm_cmp_ps_mask_neq_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_neq_oq
  // CHECK: fcmp one <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm_cmp_ps_mask_ge_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ge_os
  // CHECK: fcmp oge <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_GE_OS);
}

__mmask8 test_mm_cmp_ps_mask_gt_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_gt_os
  // CHECK: fcmp ogt <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_GT_OS);
}

__mmask8 test_mm_cmp_ps_mask_true_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_true_uq
  // CHECK: fcmp true <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm_cmp_ps_mask_eq_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_eq_os
  // CHECK: fcmp oeq <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_EQ_OS);
}

__mmask8 test_mm_cmp_ps_mask_lt_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_lt_oq
  // CHECK: fcmp olt <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_LT_OQ);
}

__mmask8 test_mm_cmp_ps_mask_le_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_le_oq
  // CHECK: fcmp ole <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_LE_OQ);
}

__mmask8 test_mm_cmp_ps_mask_unord_s(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_unord_s
  // CHECK: fcmp uno <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_UNORD_S);
}

__mmask8 test_mm_cmp_ps_mask_neq_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_neq_us
  // CHECK: fcmp une <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_NEQ_US);
}

__mmask8 test_mm_cmp_ps_mask_nlt_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nlt_uq
  // CHECK: fcmp uge <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm_cmp_ps_mask_nle_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nle_uq
  // CHECK: fcmp ugt <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm_cmp_ps_mask_ord_s(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ord_s
  // CHECK: fcmp ord <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_ORD_S);
}

__mmask8 test_mm_cmp_ps_mask_eq_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_eq_us
  // CHECK: fcmp ueq <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_EQ_US);
}

__mmask8 test_mm_cmp_ps_mask_nge_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_nge_uq
  // CHECK: fcmp ult <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm_cmp_ps_mask_ngt_uq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ngt_uq
  // CHECK: fcmp ule <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm_cmp_ps_mask_false_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_false_os
  // CHECK: fcmp false <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm_cmp_ps_mask_neq_os(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_neq_os
  // CHECK: fcmp one <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm_cmp_ps_mask_ge_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_ge_oq
  // CHECK: fcmp oge <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_GE_OQ);
}

__mmask8 test_mm_cmp_ps_mask_gt_oq(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_gt_oq
  // CHECK: fcmp ogt <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_GT_OQ);
}

__mmask8 test_mm_cmp_ps_mask_true_us(__m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_cmp_ps_mask_true_us
  // CHECK: fcmp true <4 x float> %{{.*}}, %{{.*}}
  return _mm_cmp_ps_mask(a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_eq_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: @test_mm_mask_cmp_ps_mask_eq_oq
  // CHECK: [[CMP:%.*]] = fcmp oeq <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_lt_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_lt_os
  // CHECK: [[CMP:%.*]] = fcmp olt <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_LT_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_le_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_le_os
  // CHECK: [[CMP:%.*]] = fcmp ole <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_LE_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_unord_q(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_unord_q
  // CHECK: [[CMP:%.*]] = fcmp uno <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm_mask_cmp_ps_mask_neq_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_neq_uq
  // CHECK: [[CMP:%.*]] = fcmp une <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_nlt_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nlt_us
  // CHECK: [[CMP:%.*]] = fcmp uge <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NLT_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_nle_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nle_us
  // CHECK: [[CMP:%.*]] = fcmp ugt <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NLE_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_ord_q(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ord_q
  // CHECK: [[CMP:%.*]] = fcmp ord <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_ORD_Q);
}

__mmask8 test_mm_mask_cmp_ps_mask_eq_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_eq_uq
  // CHECK: [[CMP:%.*]] = fcmp ueq <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_nge_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nge_us
  // CHECK: [[CMP:%.*]] = fcmp ult <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NGE_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_ngt_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ngt_us
  // CHECK: [[CMP:%.*]] = fcmp ule <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NGT_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_false_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_false_oq
  // CHECK: [[CMP:%.*]] = fcmp false <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_neq_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_neq_oq
  // CHECK: [[CMP:%.*]] = fcmp one <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_ge_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ge_os
  // CHECK: [[CMP:%.*]] = fcmp oge <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_GE_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_gt_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_gt_os
  // CHECK: [[CMP:%.*]] = fcmp ogt <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_GT_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_true_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_true_uq
  // CHECK: [[CMP:%.*]] = fcmp true <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_eq_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_eq_os
  // CHECK: [[CMP:%.*]] = fcmp oeq <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_EQ_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_lt_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_lt_oq
  // CHECK: [[CMP:%.*]] = fcmp olt <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_LT_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_le_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_le_oq
  // CHECK: [[CMP:%.*]] = fcmp ole <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_LE_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_unord_s(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_unord_s
  // CHECK: [[CMP:%.*]] = fcmp uno <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_UNORD_S);
}

__mmask8 test_mm_mask_cmp_ps_mask_neq_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_neq_us
  // CHECK: [[CMP:%.*]] = fcmp une <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_nlt_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nlt_uq
  // CHECK: [[CMP:%.*]] = fcmp uge <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_nle_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nle_uq
  // CHECK: [[CMP:%.*]] = fcmp ugt <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_ord_s(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ord_s
  // CHECK: [[CMP:%.*]] = fcmp ord <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_ORD_S);
}

__mmask8 test_mm_mask_cmp_ps_mask_eq_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_eq_us
  // CHECK: [[CMP:%.*]] = fcmp ueq <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_EQ_US);
}

__mmask8 test_mm_mask_cmp_ps_mask_nge_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_nge_uq
  // CHECK: [[CMP:%.*]] = fcmp ult <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_ngt_uq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ngt_uq
  // CHECK: [[CMP:%.*]] = fcmp ule <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_false_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_false_os
  // CHECK: [[CMP:%.*]] = fcmp false <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_neq_os(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_neq_os
  // CHECK: [[CMP:%.*]] = fcmp one <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm_mask_cmp_ps_mask_ge_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_ge_oq
  // CHECK: [[CMP:%.*]] = fcmp oge <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_GE_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_gt_oq(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_gt_oq
  // CHECK: [[CMP:%.*]] = fcmp ogt <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_GT_OQ);
}

__mmask8 test_mm_mask_cmp_ps_mask_true_us(__mmask8 m, __m128 a, __m128 b) {
  // CHECK-LABEL: test_mm_mask_cmp_ps_mask_true_us
  // CHECK: [[CMP:%.*]] = fcmp true <4 x float> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ps_mask(m, a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_cmp_pd_mask_eq_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: @test_mm_cmp_pd_mask_eq_oq
  // CHECK: fcmp oeq <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm_cmp_pd_mask_lt_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_lt_os
  // CHECK: fcmp olt <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm_cmp_pd_mask_le_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_le_os
  // CHECK: fcmp ole <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_LE_OS);
}

__mmask8 test_mm_cmp_pd_mask_unord_q(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_unord_q
  // CHECK: fcmp uno <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm_cmp_pd_mask_neq_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_neq_uq
  // CHECK: fcmp une <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm_cmp_pd_mask_nlt_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nlt_us
  // CHECK: fcmp uge <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_NLT_US);
}

__mmask8 test_mm_cmp_pd_mask_nle_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nle_us
  // CHECK: fcmp ugt <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_NLE_US);
}

__mmask8 test_mm_cmp_pd_mask_ord_q(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ord_q
  // CHECK: fcmp ord <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_ORD_Q);
}

__mmask8 test_mm_cmp_pd_mask_eq_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_eq_uq
  // CHECK: fcmp ueq <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm_cmp_pd_mask_nge_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nge_us
  // CHECK: fcmp ult <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_NGE_US);
}

__mmask8 test_mm_cmp_pd_mask_ngt_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ngt_us
  // CHECK: fcmp ule <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_NGT_US);
}

__mmask8 test_mm_cmp_pd_mask_false_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_false_oq
  // CHECK: fcmp false <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm_cmp_pd_mask_neq_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_neq_oq
  // CHECK: fcmp one <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm_cmp_pd_mask_ge_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ge_os
  // CHECK: fcmp oge <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_GE_OS);
}

__mmask8 test_mm_cmp_pd_mask_gt_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_gt_os
  // CHECK: fcmp ogt <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_GT_OS);
}

__mmask8 test_mm_cmp_pd_mask_true_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_true_uq
  // CHECK: fcmp true <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm_cmp_pd_mask_eq_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_eq_os
  // CHECK: fcmp oeq <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_EQ_OS);
}

__mmask8 test_mm_cmp_pd_mask_lt_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_lt_oq
  // CHECK: fcmp olt <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_LT_OQ);
}

__mmask8 test_mm_cmp_pd_mask_le_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_le_oq
  // CHECK: fcmp ole <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_LE_OQ);
}

__mmask8 test_mm_cmp_pd_mask_unord_s(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_unord_s
  // CHECK: fcmp uno <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_UNORD_S);
}

__mmask8 test_mm_cmp_pd_mask_neq_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_neq_us
  // CHECK: fcmp une <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_NEQ_US);
}

__mmask8 test_mm_cmp_pd_mask_nlt_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nlt_uq
  // CHECK: fcmp uge <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm_cmp_pd_mask_nle_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nle_uq
  // CHECK: fcmp ugt <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm_cmp_pd_mask_ord_s(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ord_s
  // CHECK: fcmp ord <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_ORD_S);
}

__mmask8 test_mm_cmp_pd_mask_eq_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_eq_us
  // CHECK: fcmp ueq <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_EQ_US);
}

__mmask8 test_mm_cmp_pd_mask_nge_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_nge_uq
  // CHECK: fcmp ult <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm_cmp_pd_mask_ngt_uq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ngt_uq
  // CHECK: fcmp ule <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm_cmp_pd_mask_false_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_false_os
  // CHECK: fcmp false <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm_cmp_pd_mask_neq_os(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_neq_os
  // CHECK: fcmp one <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm_cmp_pd_mask_ge_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_ge_oq
  // CHECK: fcmp oge <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_GE_OQ);
}

__mmask8 test_mm_cmp_pd_mask_gt_oq(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_gt_oq
  // CHECK: fcmp ogt <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_GT_OQ);
}

__mmask8 test_mm_cmp_pd_mask_true_us(__m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_cmp_pd_mask_true_us
  // CHECK: fcmp true <2 x double> %{{.*}}, %{{.*}}
  return _mm_cmp_pd_mask(a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_eq_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: @test_mm_mask_cmp_pd_mask_eq_oq
  // CHECK: [[CMP:%.*]] = fcmp oeq <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_lt_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_lt_os
  // CHECK: [[CMP:%.*]] = fcmp olt <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_LT_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_le_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_le_os
  // CHECK: [[CMP:%.*]] = fcmp ole <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_LE_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_unord_q(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_unord_q
  // CHECK: [[CMP:%.*]] = fcmp uno <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm_mask_cmp_pd_mask_neq_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_neq_uq
  // CHECK: [[CMP:%.*]] = fcmp une <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_nlt_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nlt_us
  // CHECK: [[CMP:%.*]] = fcmp uge <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NLT_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_nle_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nle_us
  // CHECK: [[CMP:%.*]] = fcmp ugt <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NLE_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_ord_q(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ord_q
  // CHECK: [[CMP:%.*]] = fcmp ord <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_ORD_Q);
}

__mmask8 test_mm_mask_cmp_pd_mask_eq_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_eq_uq
  // CHECK: [[CMP:%.*]] = fcmp ueq <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_nge_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nge_us
  // CHECK: [[CMP:%.*]] = fcmp ult <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NGE_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_ngt_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ngt_us
  // CHECK: [[CMP:%.*]] = fcmp ule <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NGT_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_false_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_false_oq
  // CHECK: [[CMP:%.*]] = fcmp false <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_neq_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_neq_oq
  // CHECK: [[CMP:%.*]] = fcmp one <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_ge_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ge_os
  // CHECK: [[CMP:%.*]] = fcmp oge <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_GE_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_gt_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_gt_os
  // CHECK: [[CMP:%.*]] = fcmp ogt <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_GT_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_true_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_true_uq
  // CHECK: [[CMP:%.*]] = fcmp true <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_eq_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_eq_os
  // CHECK: [[CMP:%.*]] = fcmp oeq <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_EQ_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_lt_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_lt_oq
  // CHECK: [[CMP:%.*]] = fcmp olt <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_LT_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_le_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_le_oq
  // CHECK: [[CMP:%.*]] = fcmp ole <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_LE_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_unord_s(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_unord_s
  // CHECK: [[CMP:%.*]] = fcmp uno <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_UNORD_S);
}

__mmask8 test_mm_mask_cmp_pd_mask_neq_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_neq_us
  // CHECK: [[CMP:%.*]] = fcmp une <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_nlt_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nlt_uq
  // CHECK: [[CMP:%.*]] = fcmp uge <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_nle_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nle_uq
  // CHECK: [[CMP:%.*]] = fcmp ugt <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_ord_s(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ord_s
  // CHECK: [[CMP:%.*]] = fcmp ord <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_ORD_S);
}

__mmask8 test_mm_mask_cmp_pd_mask_eq_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_eq_us
  // CHECK: [[CMP:%.*]] = fcmp ueq <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_EQ_US);
}

__mmask8 test_mm_mask_cmp_pd_mask_nge_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_nge_uq
  // CHECK: [[CMP:%.*]] = fcmp ult <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_ngt_uq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ngt_uq
  // CHECK: [[CMP:%.*]] = fcmp ule <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_false_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_false_os
  // CHECK: [[CMP:%.*]] = fcmp false <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_neq_os(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_neq_os
  // CHECK: [[CMP:%.*]] = fcmp one <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm_mask_cmp_pd_mask_ge_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_ge_oq
  // CHECK: [[CMP:%.*]] = fcmp oge <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_GE_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_gt_oq(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_gt_oq
  // CHECK: [[CMP:%.*]] = fcmp ogt <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_GT_OQ);
}

__mmask8 test_mm_mask_cmp_pd_mask_true_us(__mmask8 m, __m128d a, __m128d b) {
  // CHECK-LABEL: test_mm_mask_cmp_pd_mask_true_us
  // CHECK: [[CMP:%.*]] = fcmp true <2 x double> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_pd_mask(m, a, b, _CMP_TRUE_US);
}

__m128d test_mm_mask_fmadd_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_mask_fmadd_pd
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_fmadd_pd(__A, __U, __B, __C);
}

__m128d test_mm_mask_fmsub_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_mask_fmsub_pd
  // CHECK: fneg <2 x double> %{{.*}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_fmsub_pd(__A, __U, __B, __C);
}

__m128d test_mm_mask3_fmadd_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmadd_pd
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask3_fmadd_pd(__A, __B, __C, __U);
}

__m128d test_mm_mask3_fnmadd_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fnmadd_pd
  // CHECK: fneg <2 x double> %{{.*}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask3_fnmadd_pd(__A, __B, __C, __U);
}

__m128d test_mm_maskz_fmadd_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_maskz_fmadd_pd
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_fmadd_pd(__U, __A, __B, __C);
}

__m128d test_mm_maskz_fmsub_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_maskz_fmsub_pd
  // CHECK: fneg <2 x double> %{{.*}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_fmsub_pd(__U, __A, __B, __C);
}

__m128d test_mm_maskz_fnmadd_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_maskz_fnmadd_pd
  // CHECK: fneg <2 x double> %{{.*}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_fnmadd_pd(__U, __A, __B, __C);
}

__m128d test_mm_maskz_fnmsub_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_maskz_fnmsub_pd
  // CHECK: fneg <2 x double> %{{.*}}
  // CHECK: fneg <2 x double> %{{.*}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_fnmsub_pd(__U, __A, __B, __C);
}

__m256d test_mm256_mask_fmadd_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_mask_fmadd_pd
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_fmadd_pd(__A, __U, __B, __C);
}

__m256d test_mm256_mask_fmsub_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_mask_fmsub_pd
  // CHECK: fneg <4 x double> %{{.*}}
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_fmsub_pd(__A, __U, __B, __C);
}

__m256d test_mm256_mask3_fmadd_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmadd_pd
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask3_fmadd_pd(__A, __B, __C, __U);
}

__m256d test_mm256_mask3_fnmadd_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fnmadd_pd
  // CHECK: fneg <4 x double> %{{.*}}
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask3_fnmadd_pd(__A, __B, __C, __U);
}

__m256d test_mm256_maskz_fmadd_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmadd_pd
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_fmadd_pd(__U, __A, __B, __C);
}

__m256d test_mm256_maskz_fmsub_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmsub_pd
  // CHECK: fneg <4 x double> %{{.*}}
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_fmsub_pd(__U, __A, __B, __C);
}

__m256d test_mm256_maskz_fnmadd_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_maskz_fnmadd_pd
  // CHECK: fneg <4 x double> %{{.*}}
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_fnmadd_pd(__U, __A, __B, __C);
}

__m256d test_mm256_maskz_fnmsub_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_maskz_fnmsub_pd
  // CHECK: fneg <4 x double> %{{.*}}
  // CHECK: fneg <4 x double> %{{.*}}
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_fnmsub_pd(__U, __A, __B, __C);
}

__m128 test_mm_mask_fmadd_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_mask_fmadd_ps
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_fmadd_ps(__A, __U, __B, __C);
}

__m128 test_mm_mask_fmsub_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_mask_fmsub_ps
  // CHECK: fneg <4 x float> %{{.*}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_fmsub_ps(__A, __U, __B, __C);
}

__m128 test_mm_mask3_fmadd_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmadd_ps
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask3_fmadd_ps(__A, __B, __C, __U);
}

__m128 test_mm_mask3_fnmadd_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fnmadd_ps
  // CHECK: fneg <4 x float> %{{.*}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask3_fnmadd_ps(__A, __B, __C, __U);
}

__m128 test_mm_maskz_fmadd_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_maskz_fmadd_ps
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_fmadd_ps(__U, __A, __B, __C);
}

__m128 test_mm_maskz_fmsub_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_maskz_fmsub_ps
  // CHECK: fneg <4 x float> %{{.*}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_fmsub_ps(__U, __A, __B, __C);
}

__m128 test_mm_maskz_fnmadd_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_maskz_fnmadd_ps
  // CHECK: fneg <4 x float> %{{.*}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_fnmadd_ps(__U, __A, __B, __C);
}

__m128 test_mm_maskz_fnmsub_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_maskz_fnmsub_ps
  // CHECK: fneg <4 x float> %{{.*}}
  // CHECK: fneg <4 x float> %{{.*}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_fnmsub_ps(__U, __A, __B, __C);
}

__m256 test_mm256_mask_fmadd_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_mask_fmadd_ps
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_fmadd_ps(__A, __U, __B, __C);
}

__m256 test_mm256_mask_fmsub_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_mask_fmsub_ps
  // CHECK: fneg <8 x float> %{{.*}}
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_fmsub_ps(__A, __U, __B, __C);
}

__m256 test_mm256_mask3_fmadd_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmadd_ps
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask3_fmadd_ps(__A, __B, __C, __U);
}

__m256 test_mm256_mask3_fnmadd_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fnmadd_ps
  // CHECK: fneg <8 x float> %{{.*}}
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask3_fnmadd_ps(__A, __B, __C, __U);
}

__m256 test_mm256_maskz_fmadd_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmadd_ps
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_fmadd_ps(__U, __A, __B, __C);
}

__m256 test_mm256_maskz_fmsub_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmsub_ps
  // CHECK: fneg <8 x float> %{{.*}}
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_fmsub_ps(__U, __A, __B, __C);
}

__m256 test_mm256_maskz_fnmadd_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_maskz_fnmadd_ps
  // CHECK: fneg <8 x float> %{{.*}}
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_fnmadd_ps(__U, __A, __B, __C);
}

__m256 test_mm256_maskz_fnmsub_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_maskz_fnmsub_ps
  // CHECK: fneg <8 x float> %{{.*}}
  // CHECK: fneg <8 x float> %{{.*}}
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_fnmsub_ps(__U, __A, __B, __C);
}

__m128d test_mm_mask_fmaddsub_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_mask_fmaddsub_pd
  // CHECK-NOT: fneg
  // CHECK: call <2 x double> @llvm.x86.fma.vfmaddsub.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_fmaddsub_pd(__A, __U, __B, __C);
}

__m128d test_mm_mask_fmsubadd_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_mask_fmsubadd_pd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: call <2 x double> @llvm.x86.fma.vfmaddsub.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[NEG]])
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_fmsubadd_pd(__A, __U, __B, __C);
}

__m128d test_mm_mask3_fmaddsub_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmaddsub_pd
  // CHECK-NOT: fneg
  // CHECK: call <2 x double> @llvm.x86.fma.vfmaddsub.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask3_fmaddsub_pd(__A, __B, __C, __U);
}

__m128d test_mm_maskz_fmaddsub_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_maskz_fmaddsub_pd
  // CHECK-NOT: fneg
  // CHECK: call <2 x double> @llvm.x86.fma.vfmaddsub.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_fmaddsub_pd(__U, __A, __B, __C);
}

__m128d test_mm_maskz_fmsubadd_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_maskz_fmsubadd_pd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.*}}
  // CHECK: call <2 x double> @llvm.x86.fma.vfmaddsub.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[NEG]])
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_fmsubadd_pd(__U, __A, __B, __C);
}

__m256d test_mm256_mask_fmaddsub_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_mask_fmaddsub_pd
  // CHECK-NOT: fneg
  // CHECK: call <4 x double> @llvm.x86.fma.vfmaddsub.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_fmaddsub_pd(__A, __U, __B, __C);
}

__m256d test_mm256_mask_fmsubadd_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_mask_fmsubadd_pd
  // CHECK: [[NEG:%.+]] = fneg <4 x double> %{{.*}}
  // CHECK: call <4 x double> @llvm.x86.fma.vfmaddsub.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> [[NEG]])
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_fmsubadd_pd(__A, __U, __B, __C);
}

__m256d test_mm256_mask3_fmaddsub_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmaddsub_pd
  // CHECK-NOT: fneg
  // CHECK: call <4 x double> @llvm.x86.fma.vfmaddsub.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask3_fmaddsub_pd(__A, __B, __C, __U);
}

__m256d test_mm256_maskz_fmaddsub_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmaddsub_pd
  // CHECK-NOT: fneg
  // CHECK: call <4 x double> @llvm.x86.fma.vfmaddsub.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_fmaddsub_pd(__U, __A, __B, __C);
}

__m256d test_mm256_maskz_fmsubadd_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmsubadd_pd
  // CHECK: [[NEG:%.+]] = fneg <4 x double> %{{.*}}
  // CHECK: call <4 x double> @llvm.x86.fma.vfmaddsub.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> [[NEG]])
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_fmsubadd_pd(__U, __A, __B, __C);
}

__m128 test_mm_mask_fmaddsub_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_mask_fmaddsub_ps
  // CHECK-NOT: fneg
  // CHECK: call <4 x float> @llvm.x86.fma.vfmaddsub.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_fmaddsub_ps(__A, __U, __B, __C);
}

__m128 test_mm_mask_fmsubadd_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_mask_fmsubadd_ps
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: call <4 x float> @llvm.x86.fma.vfmaddsub.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[NEG]])
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_fmsubadd_ps(__A, __U, __B, __C);
}

__m128 test_mm_mask3_fmaddsub_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmaddsub_ps
  // CHECK-NOT: fneg
  // CHECK: call <4 x float> @llvm.x86.fma.vfmaddsub.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask3_fmaddsub_ps(__A, __B, __C, __U);
}

__m128 test_mm_maskz_fmaddsub_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_maskz_fmaddsub_ps
  // CHECK-NOT: fneg
  // CHECK: call <4 x float> @llvm.x86.fma.vfmaddsub.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_fmaddsub_ps(__U, __A, __B, __C);
}

__m128 test_mm_maskz_fmsubadd_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_maskz_fmsubadd_ps
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.*}}
  // CHECK: call <4 x float> @llvm.x86.fma.vfmaddsub.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[NEG]])
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_fmsubadd_ps(__U, __A, __B, __C);
}

__m256 test_mm256_mask_fmaddsub_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_mask_fmaddsub_ps
  // CHECK-NOT: fneg
  // CHECK: call <8 x float> @llvm.x86.fma.vfmaddsub.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_fmaddsub_ps(__A, __U, __B, __C);
}

__m256 test_mm256_mask_fmsubadd_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_mask_fmsubadd_ps
  // CHECK: [[NEG:%.+]] = fneg <8 x float> %{{.*}}
  // CHECK: call <8 x float> @llvm.x86.fma.vfmaddsub.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> [[NEG]])
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_fmsubadd_ps(__A, __U, __B, __C);
}

__m256 test_mm256_mask3_fmaddsub_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmaddsub_ps
  // CHECK-NOT: fneg
  // CHECK: call <8 x float> @llvm.x86.fma.vfmaddsub.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask3_fmaddsub_ps(__A, __B, __C, __U);
}

__m256 test_mm256_maskz_fmaddsub_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmaddsub_ps
  // CHECK-NOT: fneg
  // CHECK: call <8 x float> @llvm.x86.fma.vfmaddsub.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_fmaddsub_ps(__U, __A, __B, __C);
}

__m256 test_mm256_maskz_fmsubadd_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmsubadd_ps
  // CHECK: [[NEG:%.+]] = fneg <8 x float> %{{.*}}
  // CHECK: call <8 x float> @llvm.x86.fma.vfmaddsub.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> [[NEG]])
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_fmsubadd_ps(__U, __A, __B, __C);
}

__m128d test_mm_mask3_fmsub_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmsub_pd
  // CHECK: fneg <2 x double> %{{.*}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask3_fmsub_pd(__A, __B, __C, __U);
}

__m256d test_mm256_mask3_fmsub_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmsub_pd
  // CHECK: fneg <4 x double> %{{.*}}
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask3_fmsub_pd(__A, __B, __C, __U);
}

__m128 test_mm_mask3_fmsub_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmsub_ps
  // CHECK: fneg <4 x float> %{{.*}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask3_fmsub_ps(__A, __B, __C, __U);
}

__m256 test_mm256_mask3_fmsub_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmsub_ps
  // CHECK: fneg <8 x float> %{{.*}}
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask3_fmsub_ps(__A, __B, __C, __U);
}

__m128d test_mm_mask3_fmsubadd_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmsubadd_pd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.+}}
  // CHECK: call <2 x double> @llvm.x86.fma.vfmaddsub.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[NEG]])
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask3_fmsubadd_pd(__A, __B, __C, __U);
}

__m256d test_mm256_mask3_fmsubadd_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmsubadd_pd
  // CHECK: [[NEG:%.+]] = fneg <4 x double> %{{.+}}
  // CHECK: call <4 x double> @llvm.x86.fma.vfmaddsub.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> [[NEG]])
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask3_fmsubadd_pd(__A, __B, __C, __U);
}

__m128 test_mm_mask3_fmsubadd_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmsubadd_ps
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.+}}
  // CHECK: call <4 x float> @llvm.x86.fma.vfmaddsub.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[NEG]])
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask3_fmsubadd_ps(__A, __B, __C, __U);
}

__m256 test_mm256_mask3_fmsubadd_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmsubadd_ps
  // CHECK: [[NEG:%.+]] = fneg <8 x float> %{{.*}}
  // CHECK: call <8 x float> @llvm.x86.fma.vfmaddsub.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> [[NEG]])
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask3_fmsubadd_ps(__A, __B, __C, __U);
}

__m128d test_mm_mask_fnmadd_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_mask_fnmadd_pd
  // CHECK: fneg <2 x double> %{{.*}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_fnmadd_pd(__A, __U, __B, __C);
}

__m256d test_mm256_mask_fnmadd_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_mask_fnmadd_pd
  // CHECK: fneg <4 x double> %{{.*}}
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_fnmadd_pd(__A, __U, __B, __C);
}

__m128 test_mm_mask_fnmadd_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_mask_fnmadd_ps
  // CHECK: fneg <4 x float> %{{.*}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_fnmadd_ps(__A, __U, __B, __C);
}

__m256 test_mm256_mask_fnmadd_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_mask_fnmadd_ps
  // CHECK: fneg <8 x float> %{{.*}}
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_fnmadd_ps(__A, __U, __B, __C);
}

__m128d test_mm_mask_fnmsub_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_mask_fnmsub_pd
  // CHECK: fneg <2 x double> %{{.*}}
  // CHECK: fneg <2 x double> %{{.*}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_fnmsub_pd(__A, __U, __B, __C);
}

__m128d test_mm_mask3_fnmsub_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fnmsub_pd
  // CHECK: fneg <2 x double> %{{.*}}
  // CHECK: fneg <2 x double> %{{.*}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask3_fnmsub_pd(__A, __B, __C, __U);
}

__m256d test_mm256_mask_fnmsub_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_mask_fnmsub_pd
  // CHECK: fneg <4 x double> %{{.*}}
  // CHECK: fneg <4 x double> %{{.*}}
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_fnmsub_pd(__A, __U, __B, __C);
}

__m256d test_mm256_mask3_fnmsub_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fnmsub_pd
  // CHECK: fneg <4 x double> %{{.*}}
  // CHECK: fneg <4 x double> %{{.*}}
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask3_fnmsub_pd(__A, __B, __C, __U);
}

__m128 test_mm_mask_fnmsub_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_mask_fnmsub_ps
  // CHECK: fneg <4 x float> %{{.*}}
  // CHECK: fneg <4 x float> %{{.*}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_fnmsub_ps(__A, __U, __B, __C);
}

__m128 test_mm_mask3_fnmsub_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fnmsub_ps
  // CHECK: fneg <4 x float> %{{.*}}
  // CHECK: fneg <4 x float> %{{.*}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask3_fnmsub_ps(__A, __B, __C, __U);
}

__m256 test_mm256_mask_fnmsub_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_mask_fnmsub_ps
  // CHECK: fneg <8 x float> %{{.*}}
  // CHECK: fneg <8 x float> %{{.*}}
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_fnmsub_ps(__A, __U, __B, __C);
}

__m256 test_mm256_mask3_fnmsub_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fnmsub_ps
  // CHECK: fneg <8 x float> %{{.*}}
  // CHECK: fneg <8 x float> %{{.*}}
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask3_fnmsub_ps(__A, __B, __C, __U);
}

__m128d test_mm_mask_add_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_add_pd
  // CHECK: fadd <2 x double> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_add_pd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_add_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_add_pd
  // CHECK: fadd <2 x double> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_add_pd(__U,__A,__B); 
}
__m256d test_mm256_mask_add_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_add_pd
  // CHECK: fadd <4 x double> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_add_pd(__W,__U,__A,__B); 
}
__m256d test_mm256_maskz_add_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_add_pd
  // CHECK: fadd <4 x double> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_add_pd(__U,__A,__B); 
}
__m128 test_mm_mask_add_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_add_ps
  // CHECK: fadd <4 x float> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_add_ps(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_add_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_add_ps
  // CHECK: fadd <4 x float> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_add_ps(__U,__A,__B); 
}
__m256 test_mm256_mask_add_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_add_ps
  // CHECK: fadd <8 x float> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_add_ps(__W,__U,__A,__B); 
}
__m256 test_mm256_maskz_add_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_add_ps
  // CHECK: fadd <8 x float> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_add_ps(__U,__A,__B); 
}
__m128i test_mm_mask_blend_epi32(__mmask8 __U, __m128i __A, __m128i __W) {
  // CHECK-LABEL: @test_mm_mask_blend_epi32
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_blend_epi32(__U,__A,__W); 
}
__m256i test_mm256_mask_blend_epi32(__mmask8 __U, __m256i __A, __m256i __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_epi32
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_blend_epi32(__U,__A,__W); 
}
__m128d test_mm_mask_blend_pd(__mmask8 __U, __m128d __A, __m128d __W) {
  // CHECK-LABEL: @test_mm_mask_blend_pd
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_blend_pd(__U,__A,__W); 
}
__m256d test_mm256_mask_blend_pd(__mmask8 __U, __m256d __A, __m256d __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_pd
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_blend_pd(__U,__A,__W); 
}
__m128 test_mm_mask_blend_ps(__mmask8 __U, __m128 __A, __m128 __W) {
  // CHECK-LABEL: @test_mm_mask_blend_ps
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_blend_ps(__U,__A,__W); 
}
__m256 test_mm256_mask_blend_ps(__mmask8 __U, __m256 __A, __m256 __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_ps
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_blend_ps(__U,__A,__W); 
}
__m128i test_mm_mask_blend_epi64(__mmask8 __U, __m128i __A, __m128i __W) {
  // CHECK-LABEL: @test_mm_mask_blend_epi64
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_blend_epi64(__U,__A,__W); 
}
__m256i test_mm256_mask_blend_epi64(__mmask8 __U, __m256i __A, __m256i __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_epi64
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_blend_epi64(__U,__A,__W); 
}
__m128d test_mm_mask_compress_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_compress_pd
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm_mask_compress_pd(__W,__U,__A); 
}
__m128d test_mm_maskz_compress_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_compress_pd
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm_maskz_compress_pd(__U,__A); 
}
__m256d test_mm256_mask_compress_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_compress_pd
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm256_mask_compress_pd(__W,__U,__A); 
}
__m256d test_mm256_maskz_compress_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_compress_pd
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm256_maskz_compress_pd(__U,__A); 
}
__m128i test_mm_mask_compress_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_compress_epi64
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm_mask_compress_epi64(__W,__U,__A); 
}
__m128i test_mm_maskz_compress_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_compress_epi64
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm_maskz_compress_epi64(__U,__A); 
}
__m256i test_mm256_mask_compress_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_compress_epi64
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm256_mask_compress_epi64(__W,__U,__A); 
}
__m256i test_mm256_maskz_compress_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_compress_epi64
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm256_maskz_compress_epi64(__U,__A); 
}
__m128 test_mm_mask_compress_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_compress_ps
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm_mask_compress_ps(__W,__U,__A); 
}
__m128 test_mm_maskz_compress_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_compress_ps
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm_maskz_compress_ps(__U,__A); 
}
__m256 test_mm256_mask_compress_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_compress_ps
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm256_mask_compress_ps(__W,__U,__A); 
}
__m256 test_mm256_maskz_compress_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_compress_ps
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm256_maskz_compress_ps(__U,__A); 
}
__m128i test_mm_mask_compress_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_compress_epi32
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm_mask_compress_epi32(__W,__U,__A); 
}
__m128i test_mm_maskz_compress_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_compress_epi32
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm_maskz_compress_epi32(__U,__A); 
}
__m256i test_mm256_mask_compress_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_compress_epi32
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm256_mask_compress_epi32(__W,__U,__A); 
}
__m256i test_mm256_maskz_compress_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_compress_epi32
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm256_maskz_compress_epi32(__U,__A); 
}
void test_mm_mask_compressstoreu_pd(void *__P, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_compressstoreu_pd
  // CHECK: @llvm.masked.compressstore.v2f64(<2 x double> %{{.*}}, double* %{{.*}}, <2 x i1> %{{.*}})
  return _mm_mask_compressstoreu_pd(__P,__U,__A); 
}
void test_mm256_mask_compressstoreu_pd(void *__P, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_compressstoreu_pd
  // CHECK: @llvm.masked.compressstore.v4f64(<4 x double> %{{.*}}, double* %{{.*}}, <4 x i1> %{{.*}})
  return _mm256_mask_compressstoreu_pd(__P,__U,__A); 
}
void test_mm_mask_compressstoreu_epi64(void *__P, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_compressstoreu_epi64
  // CHECK: @llvm.masked.compressstore.v2i64(<2 x i64> %{{.*}}, i64* %{{.*}}, <2 x i1> %{{.*}})
  return _mm_mask_compressstoreu_epi64(__P,__U,__A); 
}
void test_mm256_mask_compressstoreu_epi64(void *__P, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_compressstoreu_epi64
  // CHECK: @llvm.masked.compressstore.v4i64(<4 x i64> %{{.*}}, i64* %{{.*}}, <4 x i1> %{{.*}})
  return _mm256_mask_compressstoreu_epi64(__P,__U,__A); 
}
void test_mm_mask_compressstoreu_ps(void *__P, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_compressstoreu_ps
  // CHECK: @llvm.masked.compressstore.v4f32(<4 x float> %{{.*}}, float* %{{.*}}, <4 x i1> %{{.*}})
  return _mm_mask_compressstoreu_ps(__P,__U,__A); 
}
void test_mm256_mask_compressstoreu_ps(void *__P, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_compressstoreu_ps
  // CHECK: @llvm.masked.compressstore.v8f32(<8 x float> %{{.*}}, float* %{{.*}}, <8 x i1> %{{.*}})
  return _mm256_mask_compressstoreu_ps(__P,__U,__A); 
}
void test_mm_mask_compressstoreu_epi32(void *__P, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_compressstoreu_epi32
  // CHECK: @llvm.masked.compressstore.v4i32(<4 x i32> %{{.*}}, i32* %{{.*}}, <4 x i1> %{{.*}})
  return _mm_mask_compressstoreu_epi32(__P,__U,__A); 
}
void test_mm256_mask_compressstoreu_epi32(void *__P, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_compressstoreu_epi32
  // CHECK: @llvm.masked.compressstore.v8i32(<8 x i32> %{{.*}}, i32* %{{.*}}, <8 x i1> %{{.*}})
  return _mm256_mask_compressstoreu_epi32(__P,__U,__A); 
}
__m128d test_mm_mask_cvtepi32_pd(__m128d __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi32_pd
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: sitofp <2 x i32> %{{.*}} to <2 x double>
  // CHECK: select <2 x i1> {{.*}}, <2 x double> {{.*}}, <2 x double> {{.*}}
  return _mm_mask_cvtepi32_pd(__W,__U,__A);
}
__m128d test_mm_maskz_cvtepi32_pd(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi32_pd
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: sitofp <2 x i32> %{{.*}} to <2 x double>
  // CHECK: select <2 x i1> {{.*}}, <2 x double> {{.*}}, <2 x double> {{.*}}
  return _mm_maskz_cvtepi32_pd(__U,__A);
}
__m256d test_mm256_mask_cvtepi32_pd(__m256d __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi32_pd
  // CHECK: sitofp <4 x i32> %{{.*}} to <4 x double>
  // CHECK: select <4 x i1> {{.*}}, <4 x double> {{.*}}, <4 x double> {{.*}}
  return _mm256_mask_cvtepi32_pd(__W,__U,__A);
}
__m256d test_mm256_maskz_cvtepi32_pd(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi32_pd
  // CHECK: sitofp <4 x i32> %{{.*}} to <4 x double>
  // CHECK: select <4 x i1> {{.*}}, <4 x double> {{.*}}, <4 x double> {{.*}}
  return _mm256_maskz_cvtepi32_pd(__U,__A);
}
__m128 test_mm_mask_cvtepi32_ps(__m128 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi32_ps
  // CHECK: sitofp <4 x i32> %{{.*}} to <4 x float>
  // CHECK: select <4 x i1> {{.*}}, <4 x float> {{.*}}, <4 x float> {{.*}}
  return _mm_mask_cvtepi32_ps(__W,__U,__A); 
}
__m128 test_mm_maskz_cvtepi32_ps(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi32_ps
  // CHECK: sitofp <4 x i32> %{{.*}} to <4 x float>
  // CHECK: select <4 x i1> {{.*}}, <4 x float> {{.*}}, <4 x float> {{.*}}
  return _mm_maskz_cvtepi32_ps(__U,__A); 
}
__m256 test_mm256_mask_cvtepi32_ps(__m256 __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi32_ps
  // CHECK: sitofp <8 x i32> %{{.*}} to <8 x float>
  // CHECK: select <8 x i1> {{.*}}, <8 x float> {{.*}}, <8 x float> {{.*}}
  return _mm256_mask_cvtepi32_ps(__W,__U,__A); 
}
__m256 test_mm256_maskz_cvtepi32_ps(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi32_ps
  // CHECK: sitofp <8 x i32> %{{.*}} to <8 x float>
  // CHECK: select <8 x i1> {{.*}}, <8 x float> {{.*}}, <8 x float> {{.*}}
  return _mm256_maskz_cvtepi32_ps(__U,__A); 
}
__m128i test_mm_mask_cvtpd_epi32(__m128i __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_cvtpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2dq.128
  return _mm_mask_cvtpd_epi32(__W,__U,__A); 
}
__m128i test_mm_maskz_cvtpd_epi32(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2dq.128
  return _mm_maskz_cvtpd_epi32(__U,__A); 
}
__m128i test_mm256_mask_cvtpd_epi32(__m128i __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtpd_epi32
  // CHECK: @llvm.x86.avx.cvt.pd2dq.256
  // CHECK: select <4 x i1> {{.*}}, <4 x i32> {{.*}}, <4 x i32> {{.*}}
  return _mm256_mask_cvtpd_epi32(__W,__U,__A); 
}
__m128i test_mm256_maskz_cvtpd_epi32(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtpd_epi32
  // CHECK: @llvm.x86.avx.cvt.pd2dq.256
  // CHECK: select <4 x i1> {{.*}}, <4 x i32> {{.*}}, <4 x i32> {{.*}}
  return _mm256_maskz_cvtpd_epi32(__U,__A); 
}
__m128 test_mm_mask_cvtpd_ps(__m128 __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_cvtpd_ps
  // CHECK: @llvm.x86.avx512.mask.cvtpd2ps
  return _mm_mask_cvtpd_ps(__W,__U,__A); 
}
__m128 test_mm_maskz_cvtpd_ps(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtpd_ps
  // CHECK: @llvm.x86.avx512.mask.cvtpd2ps
  return _mm_maskz_cvtpd_ps(__U,__A); 
}
__m128 test_mm256_mask_cvtpd_ps(__m128 __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtpd_ps
  // CHECK: @llvm.x86.avx.cvt.pd2.ps.256
  // CHECK: select <4 x i1> {{.*}}, <4 x float> {{.*}}, <4 x float> {{.*}}
  return _mm256_mask_cvtpd_ps(__W,__U,__A); 
}
__m128 test_mm256_maskz_cvtpd_ps(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtpd_ps
  // CHECK: @llvm.x86.avx.cvt.pd2.ps.256
  // CHECK: select <4 x i1> {{.*}}, <4 x float> {{.*}}, <4 x float> {{.*}}
  return _mm256_maskz_cvtpd_ps(__U,__A); 
}
__m128i test_mm_cvtpd_epu32(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvtpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.128
  return _mm_cvtpd_epu32(__A); 
}
__m128i test_mm_mask_cvtpd_epu32(__m128i __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_cvtpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.128
  return _mm_mask_cvtpd_epu32(__W,__U,__A); 
}
__m128i test_mm_maskz_cvtpd_epu32(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.128
  return _mm_maskz_cvtpd_epu32(__U,__A); 
}
__m128i test_mm256_cvtpd_epu32(__m256d __A) {
  // CHECK-LABEL: @test_mm256_cvtpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.256
  return _mm256_cvtpd_epu32(__A); 
}
__m128i test_mm256_mask_cvtpd_epu32(__m128i __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.256
  return _mm256_mask_cvtpd_epu32(__W,__U,__A); 
}
__m128i test_mm256_maskz_cvtpd_epu32(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.256
  return _mm256_maskz_cvtpd_epu32(__U,__A); 
}
__m128i test_mm_mask_cvtps_epi32(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvtps_epi32
  // CHECK: @llvm.x86.sse2.cvtps2dq
  // CHECK: select <4 x i1> {{.*}}, <4 x i32> {{.*}}, <4 x i32> {{.*}}
  return _mm_mask_cvtps_epi32(__W,__U,__A); 
}
__m128i test_mm_maskz_cvtps_epi32(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtps_epi32
  // CHECK: @llvm.x86.sse2.cvtps2dq
  // CHECK: select <4 x i1> {{.*}}, <4 x i32> {{.*}}, <4 x i32> {{.*}}
  return _mm_maskz_cvtps_epi32(__U,__A); 
}
__m256i test_mm256_mask_cvtps_epi32(__m256i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtps_epi32
  // CHECK: @llvm.x86.avx.cvt.ps2dq.256
  // CHECK: select <8 x i1> {{.*}}, <8 x i32> {{.*}}, <8 x i32> {{.*}}
  return _mm256_mask_cvtps_epi32(__W,__U,__A); 
}
__m256i test_mm256_maskz_cvtps_epi32(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtps_epi32
  // CHECK: @llvm.x86.avx.cvt.ps2dq.256
  // CHECK: select <8 x i1> {{.*}}, <8 x i32> {{.*}}, <8 x i32> {{.*}}
  return _mm256_maskz_cvtps_epi32(__U,__A); 
}
__m128d test_mm_mask_cvtps_pd(__m128d __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvtps_pd
  // CHECK: fpext <2 x float> %{{.*}} to <2 x double>
  // CHECK: select <2 x i1> {{.*}}, <2 x double> {{.*}}, <2 x double> {{.*}}
  return _mm_mask_cvtps_pd(__W,__U,__A); 
}
__m128d test_mm_maskz_cvtps_pd(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtps_pd
  // CHECK: fpext <2 x float> %{{.*}} to <2 x double>
  // CHECK: select <2 x i1> {{.*}}, <2 x double> {{.*}}, <2 x double> {{.*}}
  return _mm_maskz_cvtps_pd(__U,__A); 
}
__m256d test_mm256_mask_cvtps_pd(__m256d __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtps_pd
  // CHECK: fpext <4 x float> %{{.*}} to <4 x double>
  // CHECK: select <4 x i1> {{.*}}, <4 x double> {{.*}}, <4 x double> {{.*}}
  return _mm256_mask_cvtps_pd(__W,__U,__A); 
}
__m256d test_mm256_maskz_cvtps_pd(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtps_pd
  // CHECK: fpext <4 x float> %{{.*}} to <4 x double>
  // CHECK: select <4 x i1> {{.*}}, <4 x double> {{.*}}, <4 x double> {{.*}}
  return _mm256_maskz_cvtps_pd(__U,__A); 
}
__m128i test_mm_cvtps_epu32(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.128
  return _mm_cvtps_epu32(__A); 
}
__m128i test_mm_mask_cvtps_epu32(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvtps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.128
  return _mm_mask_cvtps_epu32(__W,__U,__A); 
}
__m128i test_mm_maskz_cvtps_epu32(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.128
  return _mm_maskz_cvtps_epu32(__U,__A); 
}
__m256i test_mm256_cvtps_epu32(__m256 __A) {
  // CHECK-LABEL: @test_mm256_cvtps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.256
  return _mm256_cvtps_epu32(__A); 
}
__m256i test_mm256_mask_cvtps_epu32(__m256i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.256
  return _mm256_mask_cvtps_epu32(__W,__U,__A); 
}
__m256i test_mm256_maskz_cvtps_epu32(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.256
  return _mm256_maskz_cvtps_epu32(__U,__A); 
}
__m128i test_mm_mask_cvttpd_epi32(__m128i __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_cvttpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2dq.128
  return _mm_mask_cvttpd_epi32(__W,__U,__A); 
}
__m128i test_mm_maskz_cvttpd_epi32(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_cvttpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2dq.128
  return _mm_maskz_cvttpd_epi32(__U,__A); 
}
__m128i test_mm256_mask_cvttpd_epi32(__m128i __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_cvttpd_epi32
  // CHECK: @llvm.x86.avx.cvtt.pd2dq.256
  // CHECK: select <4 x i1> {{.*}}, <4 x i32> {{.*}}, <4 x i32> {{.*}}
  return _mm256_mask_cvttpd_epi32(__W,__U,__A); 
}
__m128i test_mm256_maskz_cvttpd_epi32(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvttpd_epi32
  // CHECK: @llvm.x86.avx.cvtt.pd2dq.256
  // CHECK: select <4 x i1> {{.*}}, <4 x i32> {{.*}}, <4 x i32> {{.*}}
  return _mm256_maskz_cvttpd_epi32(__U,__A); 
}
__m128i test_mm_cvttpd_epu32(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvttpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.128
  return _mm_cvttpd_epu32(__A); 
}
__m128i test_mm_mask_cvttpd_epu32(__m128i __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_cvttpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.128
  return _mm_mask_cvttpd_epu32(__W,__U,__A); 
}
__m128i test_mm_maskz_cvttpd_epu32(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_cvttpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.128
  return _mm_maskz_cvttpd_epu32(__U,__A); 
}
__m128i test_mm256_cvttpd_epu32(__m256d __A) {
  // CHECK-LABEL: @test_mm256_cvttpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.256
  return _mm256_cvttpd_epu32(__A); 
}
__m128i test_mm256_mask_cvttpd_epu32(__m128i __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_cvttpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.256
  return _mm256_mask_cvttpd_epu32(__W,__U,__A); 
}
__m128i test_mm256_maskz_cvttpd_epu32(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvttpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.256
  return _mm256_maskz_cvttpd_epu32(__U,__A); 
}
__m128i test_mm_mask_cvttps_epi32(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvttps_epi32
  // CHECK: @llvm.x86.sse2.cvttps2dq
  // CHECK: select <4 x i1> {{.*}}, <4 x i32> {{.*}}, <4 x i32> {{.*}}
  return _mm_mask_cvttps_epi32(__W,__U,__A); 
}
__m128i test_mm_maskz_cvttps_epi32(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvttps_epi32
  // CHECK: @llvm.x86.sse2.cvttps2dq
  // CHECK: select <4 x i1> {{.*}}, <4 x i32> {{.*}}, <4 x i32> {{.*}}
  return _mm_maskz_cvttps_epi32(__U,__A); 
}
__m256i test_mm256_mask_cvttps_epi32(__m256i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvttps_epi32
  // CHECK: @llvm.x86.avx.cvtt.ps2dq.256
  // CHECK: select <8 x i1> {{.*}}, <8 x i32> {{.*}}, <8 x i32> {{.*}}
  return _mm256_mask_cvttps_epi32(__W,__U,__A); 
}
__m256i test_mm256_maskz_cvttps_epi32(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvttps_epi32
  // CHECK: @llvm.x86.avx.cvtt.ps2dq.256
  // CHECK: select <8 x i1> {{.*}}, <8 x i32> {{.*}}, <8 x i32> {{.*}}
  return _mm256_maskz_cvttps_epi32(__U,__A); 
}
__m128i test_mm_cvttps_epu32(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvttps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttps2udq.128
  return _mm_cvttps_epu32(__A); 
}
__m128i test_mm_mask_cvttps_epu32(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvttps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttps2udq.128
  return _mm_mask_cvttps_epu32(__W,__U,__A); 
}
__m128i test_mm_maskz_cvttps_epu32(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvttps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttps2udq.128
  return _mm_maskz_cvttps_epu32(__U,__A); 
}
__m256i test_mm256_cvttps_epu32(__m256 __A) {
  // CHECK-LABEL: @test_mm256_cvttps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttps2udq.256
  return _mm256_cvttps_epu32(__A); 
}
__m256i test_mm256_mask_cvttps_epu32(__m256i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvttps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttps2udq.256
  return _mm256_mask_cvttps_epu32(__W,__U,__A); 
}
__m256i test_mm256_maskz_cvttps_epu32(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvttps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttps2udq.256
  return _mm256_maskz_cvttps_epu32(__U,__A); 
}
__m128d test_mm_cvtepu32_pd(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepu32_pd
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: uitofp <2 x i32> %{{.*}} to <2 x double>
  return _mm_cvtepu32_pd(__A);
}
__m128d test_mm_mask_cvtepu32_pd(__m128d __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepu32_pd
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: uitofp <2 x i32> %{{.*}} to <2 x double>
  // CHECK: select <2 x i1> {{.*}}, <2 x double> {{.*}}, <2 x double> {{.*}}
  return _mm_mask_cvtepu32_pd(__W,__U,__A);
}
__m128d test_mm_maskz_cvtepu32_pd(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu32_pd
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: uitofp <2 x i32> %{{.*}} to <2 x double>
  // CHECK: select <2 x i1> {{.*}}, <2 x double> {{.*}}, <2 x double> {{.*}}
  return _mm_maskz_cvtepu32_pd(__U,__A);
}
__m256d test_mm256_cvtepu32_pd(__m128i __A) {
  // CHECK-LABEL: @test_mm256_cvtepu32_pd
  // CHECK: uitofp <4 x i32> %{{.*}} to <4 x double>
  return _mm256_cvtepu32_pd(__A);
}
__m256d test_mm256_mask_cvtepu32_pd(__m256d __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu32_pd
  // CHECK: uitofp <4 x i32> %{{.*}} to <4 x double>
  // CHECK: select <4 x i1> {{.*}}, <4 x double> {{.*}}, <4 x double> {{.*}}
  return _mm256_mask_cvtepu32_pd(__W,__U,__A);
}
__m256d test_mm256_maskz_cvtepu32_pd(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu32_pd
  // CHECK: uitofp <4 x i32> %{{.*}} to <4 x double>
  // CHECK: select <4 x i1> {{.*}}, <4 x double> {{.*}}, <4 x double> {{.*}}
  return _mm256_maskz_cvtepu32_pd(__U,__A);
}
__m128 test_mm_cvtepu32_ps(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepu32_ps
  // CHECK: uitofp <4 x i32> %{{.*}} to <4 x float>
  return _mm_cvtepu32_ps(__A); 
}
__m128 test_mm_mask_cvtepu32_ps(__m128 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepu32_ps
  // CHECK: uitofp <4 x i32> %{{.*}} to <4 x float>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_cvtepu32_ps(__W,__U,__A); 
}
__m128 test_mm_maskz_cvtepu32_ps(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu32_ps
  // CHECK: uitofp <4 x i32> %{{.*}} to <4 x float>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_cvtepu32_ps(__U,__A); 
}
__m256 test_mm256_cvtepu32_ps(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtepu32_ps
  // CHECK: uitofp <8 x i32> %{{.*}} to <8 x float>
  return _mm256_cvtepu32_ps(__A); 
}
__m256 test_mm256_mask_cvtepu32_ps(__m256 __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu32_ps
  // CHECK: uitofp <8 x i32> %{{.*}} to <8 x float>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_cvtepu32_ps(__W,__U,__A); 
}
__m256 test_mm256_maskz_cvtepu32_ps(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu32_ps
  // CHECK: uitofp <8 x i32> %{{.*}} to <8 x float>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_cvtepu32_ps(__U,__A); 
}
__m128d test_mm_mask_div_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_div_pd
  // CHECK: fdiv <2 x double> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_div_pd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_div_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_div_pd
  // CHECK: fdiv <2 x double> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_div_pd(__U,__A,__B); 
}
__m256d test_mm256_mask_div_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_div_pd
  // CHECK: fdiv <4 x double> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_div_pd(__W,__U,__A,__B); 
}
__m256d test_mm256_maskz_div_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_div_pd
  // CHECK: fdiv <4 x double> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_div_pd(__U,__A,__B); 
}
__m128 test_mm_mask_div_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_div_ps
  // CHECK: fdiv <4 x float> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_div_ps(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_div_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_div_ps
  // CHECK: fdiv <4 x float> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_div_ps(__U,__A,__B); 
}
__m256 test_mm256_mask_div_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_div_ps
  // CHECK: fdiv <8 x float> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_div_ps(__W,__U,__A,__B); 
}
__m256 test_mm256_maskz_div_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_div_ps
  // CHECK: fdiv <8 x float> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_div_ps(__U,__A,__B); 
}
__m128d test_mm_mask_expand_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_expand_pd
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm_mask_expand_pd(__W,__U,__A); 
}
__m128d test_mm_maskz_expand_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_expand_pd
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm_maskz_expand_pd(__U,__A); 
}
__m256d test_mm256_mask_expand_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_expand_pd
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm256_mask_expand_pd(__W,__U,__A); 
}
__m256d test_mm256_maskz_expand_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_expand_pd
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm256_maskz_expand_pd(__U,__A); 
}
__m128i test_mm_mask_expand_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_expand_epi64
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm_mask_expand_epi64(__W,__U,__A); 
}
__m128i test_mm_maskz_expand_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_expand_epi64
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm_maskz_expand_epi64(__U,__A); 
}
__m256i test_mm256_mask_expand_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_expand_epi64
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm256_mask_expand_epi64(__W,__U,__A); 
}
__m256i test_mm256_maskz_expand_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_expand_epi64
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm256_maskz_expand_epi64(__U,__A); 
}
__m128d test_mm_mask_expandloadu_pd(__m128d __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_expandloadu_pd
  // CHECK: @llvm.masked.expandload.v2f64(double* %{{.*}}, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_mask_expandloadu_pd(__W,__U,__P); 
}
__m128d test_mm_maskz_expandloadu_pd(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_expandloadu_pd
  // CHECK: @llvm.masked.expandload.v2f64(double* %{{.*}}, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_maskz_expandloadu_pd(__U,__P); 
}
__m256d test_mm256_mask_expandloadu_pd(__m256d __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_expandloadu_pd
  // CHECK: @llvm.masked.expandload.v4f64(double* %{{.*}}, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_mask_expandloadu_pd(__W,__U,__P); 
}
__m256d test_mm256_maskz_expandloadu_pd(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_expandloadu_pd
  // CHECK: @llvm.masked.expandload.v4f64(double* %{{.*}}, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_maskz_expandloadu_pd(__U,__P); 
}
__m128i test_mm_mask_expandloadu_epi64(__m128i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_expandloadu_epi64
  // CHECK: @llvm.masked.expandload.v2i64(i64* %{{.*}}, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_mask_expandloadu_epi64(__W,__U,__P); 
}
__m128i test_mm_maskz_expandloadu_epi64(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_expandloadu_epi64
  // CHECK: @llvm.masked.expandload.v2i64(i64* %{{.*}}, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_maskz_expandloadu_epi64(__U,__P); 
}
__m256i test_mm256_mask_expandloadu_epi64(__m256i __W, __mmask8 __U,   void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_expandloadu_epi64
  // CHECK: @llvm.masked.expandload.v4i64(i64* %{{.*}}, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_mask_expandloadu_epi64(__W,__U,__P); 
}
__m256i test_mm256_maskz_expandloadu_epi64(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_expandloadu_epi64
  // CHECK: @llvm.masked.expandload.v4i64(i64* %{{.*}}, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_maskz_expandloadu_epi64(__U,__P); 
}
__m128 test_mm_mask_expandloadu_ps(__m128 __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_expandloadu_ps
  // CHECK: @llvm.masked.expandload.v4f32(float* %{{.*}}, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_mask_expandloadu_ps(__W,__U,__P); 
}
__m128 test_mm_maskz_expandloadu_ps(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_expandloadu_ps
  // CHECK: @llvm.masked.expandload.v4f32(float* %{{.*}}, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_maskz_expandloadu_ps(__U,__P); 
}
__m256 test_mm256_mask_expandloadu_ps(__m256 __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_expandloadu_ps
  // CHECK: @llvm.masked.expandload.v8f32(float* %{{.*}}, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_mask_expandloadu_ps(__W,__U,__P); 
}
__m256 test_mm256_maskz_expandloadu_ps(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_expandloadu_ps
  // CHECK: @llvm.masked.expandload.v8f32(float* %{{.*}}, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_maskz_expandloadu_ps(__U,__P); 
}
__m128i test_mm_mask_expandloadu_epi32(__m128i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_expandloadu_epi32
  // CHECK: @llvm.masked.expandload.v4i32(i32* %{{.*}}, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_mask_expandloadu_epi32(__W,__U,__P); 
}
__m128i test_mm_maskz_expandloadu_epi32(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_expandloadu_epi32
  // CHECK: @llvm.masked.expandload.v4i32(i32* %{{.*}}, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maskz_expandloadu_epi32(__U,__P); 
}
__m256i test_mm256_mask_expandloadu_epi32(__m256i __W, __mmask8 __U,   void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_expandloadu_epi32
  // CHECK: @llvm.masked.expandload.v8i32(i32* %{{.*}}, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_mask_expandloadu_epi32(__W,__U,__P); 
}
__m256i test_mm256_maskz_expandloadu_epi32(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_expandloadu_epi32
  // CHECK: @llvm.masked.expandload.v8i32(i32* %{{.*}}, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_maskz_expandloadu_epi32(__U,__P); 
}
__m128 test_mm_mask_expand_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_expand_ps
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm_mask_expand_ps(__W,__U,__A); 
}
__m128 test_mm_maskz_expand_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_expand_ps
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm_maskz_expand_ps(__U,__A); 
}
__m256 test_mm256_mask_expand_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_expand_ps
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm256_mask_expand_ps(__W,__U,__A); 
}
__m256 test_mm256_maskz_expand_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_expand_ps
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm256_maskz_expand_ps(__U,__A); 
}
__m128i test_mm_mask_expand_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_expand_epi32
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm_mask_expand_epi32(__W,__U,__A); 
}
__m128i test_mm_maskz_expand_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_expand_epi32
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm_maskz_expand_epi32(__U,__A); 
}
__m256i test_mm256_mask_expand_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_expand_epi32
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm256_mask_expand_epi32(__W,__U,__A); 
}
__m256i test_mm256_maskz_expand_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_expand_epi32
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm256_maskz_expand_epi32(__U,__A); 
}
__m128d test_mm_getexp_pd(__m128d __A) {
  // CHECK-LABEL: @test_mm_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.128
  return _mm_getexp_pd(__A); 
}
__m128d test_mm_mask_getexp_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.128
  return _mm_mask_getexp_pd(__W,__U,__A); 
}
__m128d test_mm_maskz_getexp_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.128
  return _mm_maskz_getexp_pd(__U,__A); 
}
__m256d test_mm256_getexp_pd(__m256d __A) {
  // CHECK-LABEL: @test_mm256_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.256
  return _mm256_getexp_pd(__A); 
}
__m256d test_mm256_mask_getexp_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.256
  return _mm256_mask_getexp_pd(__W,__U,__A); 
}
__m256d test_mm256_maskz_getexp_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.256
  return _mm256_maskz_getexp_pd(__U,__A); 
}
__m128 test_mm_getexp_ps(__m128 __A) {
  // CHECK-LABEL: @test_mm_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.128
  return _mm_getexp_ps(__A); 
}
__m128 test_mm_mask_getexp_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.128
  return _mm_mask_getexp_ps(__W,__U,__A); 
}
__m128 test_mm_maskz_getexp_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.128
  return _mm_maskz_getexp_ps(__U,__A); 
}
__m256 test_mm256_getexp_ps(__m256 __A) {
  // CHECK-LABEL: @test_mm256_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.256
  return _mm256_getexp_ps(__A); 
}
__m256 test_mm256_mask_getexp_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.256
  return _mm256_mask_getexp_ps(__W,__U,__A); 
}
__m256 test_mm256_maskz_getexp_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.256
  return _mm256_maskz_getexp_ps(__U,__A); 
}
__m128d test_mm_mask_max_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_max_pd
  // CHECK: @llvm.x86.sse2.max.pd
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_max_pd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_max_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_max_pd
  // CHECK: @llvm.x86.sse2.max.pd
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_max_pd(__U,__A,__B); 
}
__m256d test_mm256_mask_max_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_max_pd
  // CHECK: @llvm.x86.avx.max.pd.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_max_pd(__W,__U,__A,__B); 
}
__m256d test_mm256_maskz_max_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_pd
  // CHECK: @llvm.x86.avx.max.pd.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_max_pd(__U,__A,__B); 
}
__m128 test_mm_mask_max_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_max_ps
  // CHECK: @llvm.x86.sse.max.ps
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_max_ps(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_max_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_max_ps
  // CHECK: @llvm.x86.sse.max.ps
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_max_ps(__U,__A,__B); 
}
__m256 test_mm256_mask_max_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_max_ps
  // CHECK: @llvm.x86.avx.max.ps.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_max_ps(__W,__U,__A,__B); 
}
__m256 test_mm256_maskz_max_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_ps
  // CHECK: @llvm.x86.avx.max.ps.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_max_ps(__U,__A,__B); 
}
__m128d test_mm_mask_min_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_min_pd
  // CHECK: @llvm.x86.sse2.min.pd
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_min_pd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_min_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_min_pd
  // CHECK: @llvm.x86.sse2.min.pd
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_min_pd(__U,__A,__B); 
}
__m256d test_mm256_mask_min_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_min_pd
  // CHECK: @llvm.x86.avx.min.pd.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_min_pd(__W,__U,__A,__B); 
}
__m256d test_mm256_maskz_min_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_pd
  // CHECK: @llvm.x86.avx.min.pd.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_min_pd(__U,__A,__B); 
}
__m128 test_mm_mask_min_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_min_ps
  // CHECK: @llvm.x86.sse.min.ps
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_min_ps(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_min_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_min_ps
  // CHECK: @llvm.x86.sse.min.ps
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_min_ps(__U,__A,__B); 
}
__m256 test_mm256_mask_min_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_min_ps
  // CHECK: @llvm.x86.avx.min.ps.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_min_ps(__W,__U,__A,__B); 
}
__m256 test_mm256_maskz_min_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_ps
  // CHECK: @llvm.x86.avx.min.ps.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_min_ps(__U,__A,__B); 
}
__m128d test_mm_mask_mul_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_mul_pd
  // CHECK: fmul <2 x double> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_mul_pd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_mul_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_mul_pd
  // CHECK: fmul <2 x double> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_mul_pd(__U,__A,__B); 
}
__m256d test_mm256_mask_mul_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_mul_pd
  // CHECK: fmul <4 x double> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_mul_pd(__W,__U,__A,__B); 
}
__m256d test_mm256_maskz_mul_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_mul_pd
  // CHECK: fmul <4 x double> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_mul_pd(__U,__A,__B); 
}
__m128 test_mm_mask_mul_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_mul_ps
  // CHECK: fmul <4 x float> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_mul_ps(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_mul_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_mul_ps
  // CHECK: fmul <4 x float> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_mul_ps(__U,__A,__B); 
}
__m256 test_mm256_mask_mul_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_mul_ps
  // CHECK: fmul <8 x float> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_mul_ps(__W,__U,__A,__B); 
}
__m256 test_mm256_maskz_mul_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_mul_ps
  // CHECK: fmul <8 x float> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_mul_ps(__U,__A,__B); 
}
__m128i test_mm_mask_abs_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_abs_epi32
  // CHECK: [[ABS:%.*]] = call <4 x i32> @llvm.abs.v4i32(<4 x i32> %{{.*}}, i1 false)
  // CHECK: [[TMP:%.*]] = bitcast <4 x i32> [[ABS]] to <2 x i64>
  // CHECK: [[ABS:%.*]] = bitcast <2 x i64> [[TMP]] to <4 x i32>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> [[ABS]], <4 x i32> %{{.*}}
  return _mm_mask_abs_epi32(__W,__U,__A); 
}
__m128i test_mm_maskz_abs_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_abs_epi32
  // CHECK: [[ABS:%.*]] = call <4 x i32> @llvm.abs.v4i32(<4 x i32> %{{.*}}, i1 false)
  // CHECK: [[TMP:%.*]] = bitcast <4 x i32> [[ABS]] to <2 x i64>
  // CHECK: [[ABS:%.*]] = bitcast <2 x i64> [[TMP]] to <4 x i32>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> [[ABS]], <4 x i32> %{{.*}}
  return _mm_maskz_abs_epi32(__U,__A); 
}
__m256i test_mm256_mask_abs_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_abs_epi32
  // CHECK: [[ABS:%.*]] = call <8 x i32> @llvm.abs.v8i32(<8 x i32> %{{.*}}, i1 false)
  // CHECK: [[TMP:%.*]] = bitcast <8 x i32> [[ABS]] to <4 x i64>
  // CHECK: [[ABS:%.*]] = bitcast <4 x i64> [[TMP]] to <8 x i32>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> [[ABS]], <8 x i32> %{{.*}}
  return _mm256_mask_abs_epi32(__W,__U,__A); 
}
__m256i test_mm256_maskz_abs_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_abs_epi32
  // CHECK: [[ABS:%.*]] = call <8 x i32> @llvm.abs.v8i32(<8 x i32> %{{.*}}, i1 false)
  // CHECK: [[TMP:%.*]] = bitcast <8 x i32> [[ABS]] to <4 x i64>
  // CHECK: [[ABS:%.*]] = bitcast <4 x i64> [[TMP]] to <8 x i32>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> [[ABS]], <8 x i32> %{{.*}}
  return _mm256_maskz_abs_epi32(__U,__A); 
}
__m128i test_mm_abs_epi64(__m128i __A) {
  // CHECK-LABEL: @test_mm_abs_epi64
  // CHECK: [[ABS:%.*]] = call <2 x i64> @llvm.abs.v2i64(<2 x i64> %{{.*}}, i1 false)
  return _mm_abs_epi64(__A); 
}
__m128i test_mm_mask_abs_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_abs_epi64
  // CHECK: [[ABS:%.*]] = call <2 x i64> @llvm.abs.v2i64(<2 x i64> %{{.*}}, i1 false)
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> [[ABS]], <2 x i64> %{{.*}}
  return _mm_mask_abs_epi64(__W,__U,__A); 
}
__m128i test_mm_maskz_abs_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_abs_epi64
  // CHECK: [[ABS:%.*]] = call <2 x i64> @llvm.abs.v2i64(<2 x i64> %{{.*}}, i1 false)
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> [[ABS]], <2 x i64> %{{.*}}
  return _mm_maskz_abs_epi64(__U,__A); 
}
__m256i test_mm256_abs_epi64(__m256i __A) {
  // CHECK-LABEL: @test_mm256_abs_epi64
  // CHECK: [[ABS:%.*]] = call <4 x i64> @llvm.abs.v4i64(<4 x i64> %{{.*}}, i1 false)
  return _mm256_abs_epi64(__A); 
}
__m256i test_mm256_mask_abs_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_abs_epi64
  // CHECK: [[ABS:%.*]] = call <4 x i64> @llvm.abs.v4i64(<4 x i64> %{{.*}}, i1 false)
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> [[ABS]], <4 x i64> %{{.*}}
  return _mm256_mask_abs_epi64(__W,__U,__A); 
}
__m256i test_mm256_maskz_abs_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_abs_epi64
  // CHECK: [[ABS:%.*]] = call <4 x i64> @llvm.abs.v4i64(<4 x i64> %{{.*}}, i1 false)
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> [[ABS]], <4 x i64> %{{.*}}
  return _mm256_maskz_abs_epi64(__U,__A); 
}
__m128i test_mm_maskz_max_epi32(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_max_epi32
  // CHECK: [[RES:%.*]] = call <4 x i32> @llvm.smax.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast <4 x i32> [[RES]] to <2 x i64>
  // CHECK: [[RES:%.*]] = bitcast <2 x i64> [[TMP]] to <4 x i32>
  // CHECK:       select <4 x i1> {{.*}}, <4 x i32> [[RES]], <4 x i32> {{.*}}
  return _mm_maskz_max_epi32(__M,__A,__B); 
}
__m128i test_mm_mask_max_epi32(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_max_epi32
  // CHECK: [[RES:%.*]] = call <4 x i32> @llvm.smax.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast <4 x i32> [[RES]] to <2 x i64>
  // CHECK: [[RES:%.*]] = bitcast <2 x i64> [[TMP]] to <4 x i32>
  // CHECK:       select <4 x i1> {{.*}}, <4 x i32> [[RES]], <4 x i32> {{.*}}
  return _mm_mask_max_epi32(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_max_epi32(__mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_epi32
  // CHECK: [[RES:%.*]] = call <8 x i32> @llvm.smax.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast <8 x i32> [[RES]] to <4 x i64>
  // CHECK: [[RES:%.*]] = bitcast <4 x i64> [[TMP]] to <8 x i32>
  // CHECK:       select <8 x i1> {{.*}}, <8 x i32> [[RES]], <8 x i32> {{.*}}
  return _mm256_maskz_max_epi32(__M,__A,__B); 
}
__m256i test_mm256_mask_max_epi32(__m256i __W, __mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_max_epi32
  // CHECK: [[RES:%.*]] = call <8 x i32> @llvm.smax.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast <8 x i32> [[RES]] to <4 x i64>
  // CHECK: [[RES:%.*]] = bitcast <4 x i64> [[TMP]] to <8 x i32>
  // CHECK:       select <8 x i1> {{.*}}, <8 x i32> [[RES]], <8 x i32> {{.*}}
  return _mm256_mask_max_epi32(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_max_epi64(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_max_epi64
  // CHECK: [[RES:%.*]] = call <2 x i64> @llvm.smax.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK:       select <2 x i1> {{.*}}, <2 x i64> [[RES]], <2 x i64> {{.*}}
  return _mm_maskz_max_epi64(__M,__A,__B); 
}
__m128i test_mm_mask_max_epi64(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_max_epi64
  // CHECK: [[RES:%.*]] = call <2 x i64> @llvm.smax.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK:       select <2 x i1> {{.*}}, <2 x i64> [[RES]], <2 x i64> {{.*}}
  return _mm_mask_max_epi64(__W,__M,__A,__B); 
}
__m128i test_mm_max_epi64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_max_epi64
  // CHECK: [[RES:%.*]] = call <2 x i64> @llvm.smax.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_max_epi64(__A,__B); 
}
__m256i test_mm256_maskz_max_epi64(__mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_epi64
  // CHECK: [[RES:%.*]] = call <4 x i64> @llvm.smax.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK:       select <4 x i1> {{.*}}, <4 x i64> [[RES]], <4 x i64> {{.*}}
  return _mm256_maskz_max_epi64(__M,__A,__B); 
}
__m256i test_mm256_mask_max_epi64(__m256i __W, __mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_max_epi64
  // CHECK: [[RES:%.*]] = call <4 x i64> @llvm.smax.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK:       select <4 x i1> {{.*}}, <4 x i64> [[RES]], <4 x i64> {{.*}}
  return _mm256_mask_max_epi64(__W,__M,__A,__B); 
}
__m256i test_mm256_max_epi64(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_max_epi64
  // CHECK: [[RES:%.*]] = call <4 x i64> @llvm.smax.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_max_epi64(__A,__B); 
}
__m128i test_mm_maskz_max_epu32(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_max_epu32
  // CHECK: [[RES:%.*]] = call <4 x i32> @llvm.umax.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast <4 x i32> [[RES]] to <2 x i64>
  // CHECK: [[RES:%.*]] = bitcast <2 x i64> [[TMP]] to <4 x i32>
  // CHECK:       select <4 x i1> {{.*}}, <4 x i32> [[RES]], <4 x i32> {{.*}}
  return _mm_maskz_max_epu32(__M,__A,__B); 
}
__m128i test_mm_mask_max_epu32(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_max_epu32
  // CHECK: [[RES:%.*]] = call <4 x i32> @llvm.umax.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast <4 x i32> [[RES]] to <2 x i64>
  // CHECK: [[RES:%.*]] = bitcast <2 x i64> [[TMP]] to <4 x i32>
  // CHECK:       select <4 x i1> {{.*}}, <4 x i32> [[RES]], <4 x i32> {{.*}}
  return _mm_mask_max_epu32(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_max_epu32(__mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_epu32
  // CHECK: [[RES:%.*]] = call <8 x i32> @llvm.umax.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast <8 x i32> [[RES]] to <4 x i64>
  // CHECK: [[RES:%.*]] = bitcast <4 x i64> [[TMP]] to <8 x i32>
  // CHECK:       select <8 x i1> {{.*}}, <8 x i32> [[RES]], <8 x i32> {{.*}}
  return _mm256_maskz_max_epu32(__M,__A,__B); 
}
__m256i test_mm256_mask_max_epu32(__m256i __W, __mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_max_epu32
  // CHECK: [[RES:%.*]] = call <8 x i32> @llvm.umax.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast <8 x i32> [[RES]] to <4 x i64>
  // CHECK: [[RES:%.*]] = bitcast <4 x i64> [[TMP]] to <8 x i32>
  // CHECK:       select <8 x i1> {{.*}}, <8 x i32> [[RES]], <8 x i32> {{.*}}
  return _mm256_mask_max_epu32(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_max_epu64(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_max_epu64
  // CHECK: [[RES:%.*]] = call <2 x i64> @llvm.umax.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK:       select <2 x i1> {{.*}}, <2 x i64> [[RES]], <2 x i64> {{.*}}
  return _mm_maskz_max_epu64(__M,__A,__B); 
}
__m128i test_mm_max_epu64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_max_epu64
  // CHECK: [[RES:%.*]] = call <2 x i64> @llvm.umax.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_max_epu64(__A,__B); 
}
__m128i test_mm_mask_max_epu64(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_max_epu64
  // CHECK: [[RES:%.*]] = call <2 x i64> @llvm.umax.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK:       select <2 x i1> {{.*}}, <2 x i64> [[RES]], <2 x i64> {{.*}}
  return _mm_mask_max_epu64(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_max_epu64(__mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_epu64
  // CHECK: [[RES:%.*]] = call <4 x i64> @llvm.umax.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK:       select <4 x i1> {{.*}}, <4 x i64> [[RES]], <4 x i64> {{.*}}
  return _mm256_maskz_max_epu64(__M,__A,__B); 
}
__m256i test_mm256_max_epu64(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_max_epu64
  // CHECK: [[RES:%.*]] = call <4 x i64> @llvm.umax.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_max_epu64(__A,__B); 
}
__m256i test_mm256_mask_max_epu64(__m256i __W, __mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_max_epu64
  // CHECK: [[RES:%.*]] = call <4 x i64> @llvm.umax.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK:       select <4 x i1> {{.*}}, <4 x i64> [[RES]], <4 x i64> {{.*}}
  return _mm256_mask_max_epu64(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_min_epi32(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_min_epi32
  // CHECK: [[RES:%.*]] = call <4 x i32> @llvm.smin.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast <4 x i32> [[RES]] to <2 x i64>
  // CHECK: [[RES:%.*]] = bitcast <2 x i64> [[TMP]] to <4 x i32>
  // CHECK:       select <4 x i1> {{.*}}, <4 x i32> [[RES]], <4 x i32> {{.*}}
  return _mm_maskz_min_epi32(__M,__A,__B); 
}
__m128i test_mm_mask_min_epi32(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_min_epi32
  // CHECK: [[RES:%.*]] = call <4 x i32> @llvm.smin.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast <4 x i32> [[RES]] to <2 x i64>
  // CHECK: [[RES:%.*]] = bitcast <2 x i64> [[TMP]] to <4 x i32>
  // CHECK:       select <4 x i1> {{.*}}, <4 x i32> [[RES]], <4 x i32> {{.*}}
  return _mm_mask_min_epi32(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_min_epi32(__mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_epi32
  // CHECK: [[RES:%.*]] = call <8 x i32> @llvm.smin.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast <8 x i32> [[RES]] to <4 x i64>
  // CHECK: [[RES:%.*]] = bitcast <4 x i64> [[TMP]] to <8 x i32>
  // CHECK:       select <8 x i1> {{.*}}, <8 x i32> [[RES]], <8 x i32> {{.*}}
  return _mm256_maskz_min_epi32(__M,__A,__B); 
}
__m256i test_mm256_mask_min_epi32(__m256i __W, __mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_min_epi32
  // CHECK: [[RES:%.*]] = call <8 x i32> @llvm.smin.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast <8 x i32> [[RES]] to <4 x i64>
  // CHECK: [[RES:%.*]] = bitcast <4 x i64> [[TMP]] to <8 x i32>
  // CHECK:       select <8 x i1> {{.*}}, <8 x i32> [[RES]], <8 x i32> {{.*}}
  return _mm256_mask_min_epi32(__W,__M,__A,__B); 
}
__m128i test_mm_min_epi64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_min_epi64
  // CHECK: [[RES:%.*]] = call <2 x i64> @llvm.smin.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_min_epi64(__A,__B); 
}
__m128i test_mm_mask_min_epi64(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_min_epi64
  // CHECK: [[RES:%.*]] = call <2 x i64> @llvm.smin.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK:       select <2 x i1> {{.*}}, <2 x i64> [[RES]], <2 x i64> {{.*}}
  return _mm_mask_min_epi64(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_min_epi64(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_min_epi64
  // CHECK: [[RES:%.*]] = call <2 x i64> @llvm.smin.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK:       select <2 x i1> {{.*}}, <2 x i64> [[RES]], <2 x i64> {{.*}}
  return _mm_maskz_min_epi64(__M,__A,__B); 
}
__m256i test_mm256_min_epi64(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_min_epi64
  // CHECK: [[RES:%.*]] = call <4 x i64> @llvm.smin.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_min_epi64(__A,__B); 
}
__m256i test_mm256_mask_min_epi64(__m256i __W, __mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_min_epi64
  // CHECK: [[RES:%.*]] = call <4 x i64> @llvm.smin.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK:       select <4 x i1> {{.*}}, <4 x i64> [[RES]], <4 x i64> {{.*}}
  return _mm256_mask_min_epi64(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_min_epi64(__mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_epi64
  // CHECK: [[RES:%.*]] = call <4 x i64> @llvm.smin.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK:       select <4 x i1> {{.*}}, <4 x i64> [[RES]], <4 x i64> {{.*}}
  return _mm256_maskz_min_epi64(__M,__A,__B); 
}
__m128i test_mm_maskz_min_epu32(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_min_epu32
  // CHECK: [[RES:%.*]] = call <4 x i32> @llvm.umin.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast <4 x i32> [[RES]] to <2 x i64>
  // CHECK: [[RES:%.*]] = bitcast <2 x i64> [[TMP]] to <4 x i32>
  // CHECK:       select <4 x i1> {{.*}}, <4 x i32> [[RES]], <4 x i32> {{.*}}
  return _mm_maskz_min_epu32(__M,__A,__B); 
}
__m128i test_mm_mask_min_epu32(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_min_epu32
  // CHECK: [[RES:%.*]] = call <4 x i32> @llvm.umin.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast <4 x i32> [[RES]] to <2 x i64>
  // CHECK: [[RES:%.*]] = bitcast <2 x i64> [[TMP]] to <4 x i32>
  // CHECK:       select <4 x i1> {{.*}}, <4 x i32> [[RES]], <4 x i32> {{.*}}
  return _mm_mask_min_epu32(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_min_epu32(__mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_epu32
  // CHECK: [[RES:%.*]] = call <8 x i32> @llvm.umin.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast <8 x i32> [[RES]] to <4 x i64>
  // CHECK: [[RES:%.*]] = bitcast <4 x i64> [[TMP]] to <8 x i32>
  // CHECK:       select <8 x i1> {{.*}}, <8 x i32> [[RES]], <8 x i32> {{.*}}
  return _mm256_maskz_min_epu32(__M,__A,__B); 
}
__m256i test_mm256_mask_min_epu32(__m256i __W, __mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_min_epu32
  // CHECK: [[RES:%.*]] = call <8 x i32> @llvm.umin.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast <8 x i32> [[RES]] to <4 x i64>
  // CHECK: [[RES:%.*]] = bitcast <4 x i64> [[TMP]] to <8 x i32>
  // CHECK:       select <8 x i1> {{.*}}, <8 x i32> [[RES]], <8 x i32> {{.*}}
  return _mm256_mask_min_epu32(__W,__M,__A,__B); 
}
__m128i test_mm_min_epu64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_min_epu64
  // CHECK: [[RES:%.*]] = call <2 x i64> @llvm.umin.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_min_epu64(__A,__B); 
}
__m128i test_mm_mask_min_epu64(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_min_epu64
  // CHECK: [[RES:%.*]] = call <2 x i64> @llvm.umin.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK:       select <2 x i1> {{.*}}, <2 x i64> [[RES]], <2 x i64> {{.*}}
  return _mm_mask_min_epu64(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_min_epu64(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_min_epu64
  // CHECK: [[RES:%.*]] = call <2 x i64> @llvm.umin.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK:       select <2 x i1> {{.*}}, <2 x i64> [[RES]], <2 x i64> {{.*}}
  return _mm_maskz_min_epu64(__M,__A,__B); 
}
__m256i test_mm256_min_epu64(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_min_epu64
  // CHECK: [[RES:%.*]] = call <4 x i64> @llvm.umin.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_min_epu64(__A,__B); 
}
__m256i test_mm256_mask_min_epu64(__m256i __W, __mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_min_epu64
  // CHECK: [[RES:%.*]] = call <4 x i64> @llvm.umin.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK:       select <4 x i1> {{.*}}, <4 x i64> [[RES]], <4 x i64> {{.*}}
  return _mm256_mask_min_epu64(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_min_epu64(__mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_epu64
  // CHECK: [[RES:%.*]] = call <4 x i64> @llvm.umin.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK:       select <4 x i1> {{.*}}, <4 x i64> [[RES]], <4 x i64> {{.*}}
  return _mm256_maskz_min_epu64(__M,__A,__B); 
}
__m128d test_mm_roundscale_pd(__m128d __A) {
  // CHECK-LABEL: @test_mm_roundscale_pd
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.128
  return _mm_roundscale_pd(__A,4); 
}
__m128d test_mm_mask_roundscale_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_roundscale_pd
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.128
  return _mm_mask_roundscale_pd(__W,__U,__A,4); 
}
__m128d test_mm_maskz_roundscale_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_roundscale_pd
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.128
  return _mm_maskz_roundscale_pd(__U,__A,4); 
}
__m256d test_mm256_roundscale_pd(__m256d __A) {
  // CHECK-LABEL: @test_mm256_roundscale_pd
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.256
  return _mm256_roundscale_pd(__A,4); 
}
__m256d test_mm256_mask_roundscale_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_roundscale_pd
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.256
  return _mm256_mask_roundscale_pd(__W,__U,__A,4); 
}
__m256d test_mm256_maskz_roundscale_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_roundscale_pd
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.256
  return _mm256_maskz_roundscale_pd(__U,__A,4); 
}
__m128 test_mm_roundscale_ps(__m128 __A) {
  // CHECK-LABEL: @test_mm_roundscale_ps
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.128
  return _mm_roundscale_ps(__A,4); 
}
__m128 test_mm_mask_roundscale_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_roundscale_ps
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.128
  return _mm_mask_roundscale_ps(__W,__U,__A,4); 
}
__m128 test_mm_maskz_roundscale_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_roundscale_ps
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.128
  return _mm_maskz_roundscale_ps(__U,__A, 4); 
}
__m256 test_mm256_roundscale_ps(__m256 __A) {
  // CHECK-LABEL: @test_mm256_roundscale_ps
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.256
  return _mm256_roundscale_ps(__A,4); 
}
__m256 test_mm256_mask_roundscale_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_roundscale_ps
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.256
  return _mm256_mask_roundscale_ps(__W,__U,__A,4); 
}
__m256 test_mm256_maskz_roundscale_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_roundscale_ps
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.256
  return _mm256_maskz_roundscale_ps(__U,__A,4); 
}
__m128d test_mm_scalef_pd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.128
  return _mm_scalef_pd(__A,__B); 
}
__m128d test_mm_mask_scalef_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.128
  return _mm_mask_scalef_pd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_scalef_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.128
  return _mm_maskz_scalef_pd(__U,__A,__B); 
}
__m256d test_mm256_scalef_pd(__m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.256
  return _mm256_scalef_pd(__A,__B); 
}
__m256d test_mm256_mask_scalef_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.256
  return _mm256_mask_scalef_pd(__W,__U,__A,__B); 
}
__m256d test_mm256_maskz_scalef_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.256
  return _mm256_maskz_scalef_pd(__U,__A,__B); 
}
__m128 test_mm_scalef_ps(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.128
  return _mm_scalef_ps(__A,__B); 
}
__m128 test_mm_mask_scalef_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.128
  return _mm_mask_scalef_ps(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_scalef_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.128
  return _mm_maskz_scalef_ps(__U,__A,__B); 
}
__m256 test_mm256_scalef_ps(__m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.256
  return _mm256_scalef_ps(__A,__B); 
}
__m256 test_mm256_mask_scalef_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.256
  return _mm256_mask_scalef_ps(__W,__U,__A,__B); 
}
__m256 test_mm256_maskz_scalef_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.256
  return _mm256_maskz_scalef_ps(__U,__A,__B); 
}
void test_mm_i64scatter_pd(double *__addr, __m128i __index,  __m128d __v1) {
  // CHECK-LABEL: @test_mm_i64scatter_pd
  // CHECK: @llvm.x86.avx512.mask.scatterdiv2.df
  return _mm_i64scatter_pd(__addr,__index,__v1,2); 
}
void test_mm_mask_i64scatter_pd(double *__addr, __mmask8 __mask, __m128i __index, __m128d __v1) {
  // CHECK-LABEL: @test_mm_mask_i64scatter_pd
  // CHECK: @llvm.x86.avx512.mask.scatterdiv2.df
  return _mm_mask_i64scatter_pd(__addr,__mask,__index,__v1,2); 
}
void test_mm_i64scatter_epi64(long long *__addr, __m128i __index,  __m128i __v1) {
  // CHECK-LABEL: @test_mm_i64scatter_epi64
  // CHECK: @llvm.x86.avx512.mask.scatterdiv2.di
  return _mm_i64scatter_epi64(__addr,__index,__v1,2); 
}
void test_mm_mask_i64scatter_epi64(long long *__addr, __mmask8 __mask, __m128i __index, __m128i __v1) {
  // CHECK-LABEL: @test_mm_mask_i64scatter_epi64
  // CHECK: @llvm.x86.avx512.mask.scatterdiv2.di
  return _mm_mask_i64scatter_epi64(__addr,__mask,__index,__v1,2); 
}
void test_mm256_i64scatter_pd(double *__addr, __m256i __index,  __m256d __v1) {
  // CHECK-LABEL: @test_mm256_i64scatter_pd
  // CHECK: @llvm.x86.avx512.mask.scatterdiv4.df
  return _mm256_i64scatter_pd(__addr,__index,__v1,2); 
}
void test_mm256_mask_i64scatter_pd(double *__addr, __mmask8 __mask, __m256i __index, __m256d __v1) {
  // CHECK-LABEL: @test_mm256_mask_i64scatter_pd
  // CHECK: @llvm.x86.avx512.mask.scatterdiv4.df
  return _mm256_mask_i64scatter_pd(__addr,__mask,__index,__v1,2); 
}
void test_mm256_i64scatter_epi64(long long *__addr, __m256i __index,  __m256i __v1) {
  // CHECK-LABEL: @test_mm256_i64scatter_epi64
  // CHECK: @llvm.x86.avx512.mask.scatterdiv4.di
  return _mm256_i64scatter_epi64(__addr,__index,__v1,2); 
}
void test_mm256_mask_i64scatter_epi64(long long *__addr, __mmask8 __mask,  __m256i __index, __m256i __v1) {
  // CHECK-LABEL: @test_mm256_mask_i64scatter_epi64
  // CHECK: @llvm.x86.avx512.mask.scatterdiv4.di
  return _mm256_mask_i64scatter_epi64(__addr,__mask,__index,__v1,2); 
}
void test_mm_i64scatter_ps(float *__addr, __m128i __index, __m128 __v1) {
  // CHECK-LABEL: @test_mm_i64scatter_ps
  // CHECK: @llvm.x86.avx512.mask.scatterdiv4.sf
  return _mm_i64scatter_ps(__addr,__index,__v1,2); 
}
void test_mm_mask_i64scatter_ps(float *__addr, __mmask8 __mask, __m128i __index, __m128 __v1) {
  // CHECK-LABEL: @test_mm_mask_i64scatter_ps
  // CHECK: @llvm.x86.avx512.mask.scatterdiv4.sf
  return _mm_mask_i64scatter_ps(__addr,__mask,__index,__v1,2); 
}
void test_mm_i64scatter_epi32(int *__addr, __m128i __index,  __m128i __v1) {
  // CHECK-LABEL: @test_mm_i64scatter_epi32
  // CHECK: @llvm.x86.avx512.mask.scatterdiv4.si
  return _mm_i64scatter_epi32(__addr,__index,__v1,2); 
}
void test_mm_mask_i64scatter_epi32(int *__addr, __mmask8 __mask, __m128i __index, __m128i __v1) {
  // CHECK-LABEL: @test_mm_mask_i64scatter_epi32
  // CHECK: @llvm.x86.avx512.mask.scatterdiv4.si
  return _mm_mask_i64scatter_epi32(__addr,__mask,__index,__v1,2); 
}
void test_mm256_i64scatter_ps(float *__addr, __m256i __index,  __m128 __v1) {
  // CHECK-LABEL: @test_mm256_i64scatter_ps
  // CHECK: @llvm.x86.avx512.mask.scatterdiv8.sf
  return _mm256_i64scatter_ps(__addr,__index,__v1,2); 
}
void test_mm256_mask_i64scatter_ps(float *__addr, __mmask8 __mask, __m256i __index, __m128 __v1) {
  // CHECK-LABEL: @test_mm256_mask_i64scatter_ps
  // CHECK: @llvm.x86.avx512.mask.scatterdiv8.sf
  return _mm256_mask_i64scatter_ps(__addr,__mask,__index,__v1,2); 
}
void test_mm256_i64scatter_epi32(int *__addr, __m256i __index,  __m128i __v1) {
  // CHECK-LABEL: @test_mm256_i64scatter_epi32
  // CHECK: @llvm.x86.avx512.mask.scatterdiv8.si
  return _mm256_i64scatter_epi32(__addr,__index,__v1,2); 
}
void test_mm256_mask_i64scatter_epi32(int *__addr, __mmask8 __mask,  __m256i __index, __m128i __v1) {
  // CHECK-LABEL: @test_mm256_mask_i64scatter_epi32
  // CHECK: @llvm.x86.avx512.mask.scatterdiv8.si
  return _mm256_mask_i64scatter_epi32(__addr,__mask,__index,__v1,2); 
}
void test_mm_i32scatter_pd(double *__addr, __m128i __index,  __m128d __v1) {
  // CHECK-LABEL: @test_mm_i32scatter_pd
  // CHECK: @llvm.x86.avx512.mask.scattersiv2.df
  return _mm_i32scatter_pd(__addr,__index,__v1,2); 
}
void test_mm_mask_i32scatter_pd(double *__addr, __mmask8 __mask, __m128i __index, __m128d __v1) {
  // CHECK-LABEL: @test_mm_mask_i32scatter_pd
  // CHECK: @llvm.x86.avx512.mask.scattersiv2.df
  return _mm_mask_i32scatter_pd(__addr,__mask,__index,__v1,2); 
}
void test_mm_i32scatter_epi64(long long *__addr, __m128i __index,  __m128i __v1) {
  // CHECK-LABEL: @test_mm_i32scatter_epi64
  // CHECK: @llvm.x86.avx512.mask.scattersiv2.di
  return _mm_i32scatter_epi64(__addr,__index,__v1,2); 
}
void test_mm_mask_i32scatter_epi64(long long *__addr, __mmask8 __mask, __m128i __index, __m128i __v1) {
  // CHECK-LABEL: @test_mm_mask_i32scatter_epi64
  // CHECK: @llvm.x86.avx512.mask.scattersiv2.di
  return _mm_mask_i32scatter_epi64(__addr,__mask,__index,__v1,2); 
}
void test_mm256_i32scatter_pd(double *__addr, __m128i __index,  __m256d __v1) {
  // CHECK-LABEL: @test_mm256_i32scatter_pd
  // CHECK: @llvm.x86.avx512.mask.scattersiv4.df
  return _mm256_i32scatter_pd(__addr,__index,__v1,2); 
}
void test_mm256_mask_i32scatter_pd(double *__addr, __mmask8 __mask, __m128i __index, __m256d __v1) {
  // CHECK-LABEL: @test_mm256_mask_i32scatter_pd
  // CHECK: @llvm.x86.avx512.mask.scattersiv4.df
  return _mm256_mask_i32scatter_pd(__addr,__mask,__index,__v1,2); 
}
void test_mm256_i32scatter_epi64(long long *__addr, __m128i __index,  __m256i __v1) {
  // CHECK-LABEL: @test_mm256_i32scatter_epi64
  // CHECK: @llvm.x86.avx512.mask.scattersiv4.di
  return _mm256_i32scatter_epi64(__addr,__index,__v1,2); 
}
void test_mm256_mask_i32scatter_epi64(long long *__addr, __mmask8 __mask,  __m128i __index, __m256i __v1) {
  // CHECK-LABEL: @test_mm256_mask_i32scatter_epi64
  // CHECK: @llvm.x86.avx512.mask.scattersiv4.di
  return _mm256_mask_i32scatter_epi64(__addr,__mask,__index,__v1,2); 
}
void test_mm_i32scatter_ps(float *__addr, __m128i __index, __m128 __v1) {
  // CHECK-LABEL: @test_mm_i32scatter_ps
  // CHECK: @llvm.x86.avx512.mask.scattersiv4.sf
  return _mm_i32scatter_ps(__addr,__index,__v1,2); 
}
void test_mm_mask_i32scatter_ps(float *__addr, __mmask8 __mask, __m128i __index, __m128 __v1) {
  // CHECK-LABEL: @test_mm_mask_i32scatter_ps
  // CHECK: @llvm.x86.avx512.mask.scattersiv4.sf
  return _mm_mask_i32scatter_ps(__addr,__mask,__index,__v1,2); 
}
void test_mm_i32scatter_epi32(int *__addr, __m128i __index,  __m128i __v1) {
  // CHECK-LABEL: @test_mm_i32scatter_epi32
  // CHECK: @llvm.x86.avx512.mask.scattersiv4.si
  return _mm_i32scatter_epi32(__addr,__index,__v1,2); 
}
void test_mm_mask_i32scatter_epi32(int *__addr, __mmask8 __mask, __m128i __index, __m128i __v1) {
  // CHECK-LABEL: @test_mm_mask_i32scatter_epi32
  // CHECK: @llvm.x86.avx512.mask.scattersiv4.si
  return _mm_mask_i32scatter_epi32(__addr,__mask,__index,__v1,2); 
}
void test_mm256_i32scatter_ps(float *__addr, __m256i __index,  __m256 __v1) {
  // CHECK-LABEL: @test_mm256_i32scatter_ps
  // CHECK: @llvm.x86.avx512.mask.scattersiv8.sf
  return _mm256_i32scatter_ps(__addr,__index,__v1,2); 
}
void test_mm256_mask_i32scatter_ps(float *__addr, __mmask8 __mask, __m256i __index, __m256 __v1) {
  // CHECK-LABEL: @test_mm256_mask_i32scatter_ps
  // CHECK: @llvm.x86.avx512.mask.scattersiv8.sf
  return _mm256_mask_i32scatter_ps(__addr,__mask,__index,__v1,2); 
}
void test_mm256_i32scatter_epi32(int *__addr, __m256i __index,  __m256i __v1) {
  // CHECK-LABEL: @test_mm256_i32scatter_epi32
  // CHECK: @llvm.x86.avx512.mask.scattersiv8.si
  return _mm256_i32scatter_epi32(__addr,__index,__v1,2); 
}
void test_mm256_mask_i32scatter_epi32(int *__addr, __mmask8 __mask,  __m256i __index, __m256i __v1) {
  // CHECK-LABEL: @test_mm256_mask_i32scatter_epi32
  // CHECK: @llvm.x86.avx512.mask.scattersiv8.si
  return _mm256_mask_i32scatter_epi32(__addr,__mask,__index,__v1,2); 
}
__m128d test_mm_mask_sqrt_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_sqrt_pd
  // CHECK: @llvm.sqrt.v2f64
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_sqrt_pd(__W,__U,__A); 
}
__m128d test_mm_maskz_sqrt_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_sqrt_pd
  // CHECK: @llvm.sqrt.v2f64
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_sqrt_pd(__U,__A); 
}
__m256d test_mm256_mask_sqrt_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_sqrt_pd
  // CHECK: @llvm.sqrt.v4f64
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_sqrt_pd(__W,__U,__A); 
}
__m256d test_mm256_maskz_sqrt_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_sqrt_pd
  // CHECK: @llvm.sqrt.v4f64
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_sqrt_pd(__U,__A); 
}
__m128 test_mm_mask_sqrt_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_sqrt_ps
  // CHECK: @llvm.sqrt.v4f32
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_sqrt_ps(__W,__U,__A); 
}
__m128 test_mm_maskz_sqrt_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_sqrt_ps
  // CHECK: @llvm.sqrt.v4f32
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_sqrt_ps(__U,__A); 
}
__m256 test_mm256_mask_sqrt_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_sqrt_ps
  // CHECK: @llvm.sqrt.v8f32
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_sqrt_ps(__W,__U,__A); 
}
__m256 test_mm256_maskz_sqrt_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_sqrt_ps
  // CHECK: @llvm.sqrt.v8f32
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_sqrt_ps(__U,__A); 
}
__m128d test_mm_mask_sub_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_sub_pd
  // CHECK: fsub <2 x double> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_sub_pd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_sub_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_sub_pd
  // CHECK: fsub <2 x double> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_sub_pd(__U,__A,__B); 
}
__m256d test_mm256_mask_sub_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_sub_pd
  // CHECK: fsub <4 x double> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_sub_pd(__W,__U,__A,__B); 
}
__m256d test_mm256_maskz_sub_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_sub_pd
  // CHECK: fsub <4 x double> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_sub_pd(__U,__A,__B); 
}
__m128 test_mm_mask_sub_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_sub_ps
  // CHECK: fsub <4 x float> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_sub_ps(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_sub_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_sub_ps
  // CHECK: fsub <4 x float> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_sub_ps(__U,__A,__B); 
}
__m256 test_mm256_mask_sub_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_sub_ps
  // CHECK: fsub <8 x float> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_sub_ps(__W,__U,__A,__B); 
}
__m256 test_mm256_maskz_sub_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_sub_ps
  // CHECK: fsub <8 x float> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_sub_ps(__U,__A,__B); 
}
__m128i test_mm_mask2_permutex2var_epi32(__m128i __A, __m128i __I, __mmask8 __U,  __m128i __B) {
  // CHECK-LABEL: @test_mm_mask2_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.vpermi2var.d.128
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask2_permutex2var_epi32(__A,__I,__U,__B); 
}
__m256i test_mm256_mask2_permutex2var_epi32(__m256i __A, __m256i __I, __mmask8 __U, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask2_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.vpermi2var.d.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask2_permutex2var_epi32(__A,__I,__U,__B); 
}
__m128d test_mm_mask2_permutex2var_pd(__m128d __A, __m128i __I, __mmask8 __U, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask2_permutex2var_pd
  // CHECK: @llvm.x86.avx512.vpermi2var.pd.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask2_permutex2var_pd(__A,__I,__U,__B); 
}
__m256d test_mm256_mask2_permutex2var_pd(__m256d __A, __m256i __I, __mmask8 __U,  __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask2_permutex2var_pd
  // CHECK: @llvm.x86.avx512.vpermi2var.pd.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask2_permutex2var_pd(__A,__I,__U,__B); 
}
__m128 test_mm_mask2_permutex2var_ps(__m128 __A, __m128i __I, __mmask8 __U, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask2_permutex2var_ps
  // CHECK: @llvm.x86.avx512.vpermi2var.ps.128
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask2_permutex2var_ps(__A,__I,__U,__B); 
}
__m256 test_mm256_mask2_permutex2var_ps(__m256 __A, __m256i __I, __mmask8 __U,  __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask2_permutex2var_ps
  // CHECK: @llvm.x86.avx512.vpermi2var.ps.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask2_permutex2var_ps(__A,__I,__U,__B); 
}
__m128i test_mm_mask2_permutex2var_epi64(__m128i __A, __m128i __I, __mmask8 __U,  __m128i __B) {
  // CHECK-LABEL: @test_mm_mask2_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.vpermi2var.q.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask2_permutex2var_epi64(__A,__I,__U,__B); 
}
__m256i test_mm256_mask2_permutex2var_epi64(__m256i __A, __m256i __I, __mmask8 __U, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask2_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.vpermi2var.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask2_permutex2var_epi64(__A,__I,__U,__B); 
}
__m128i test_mm_permutex2var_epi32(__m128i __A, __m128i __I, __m128i __B) {
  // CHECK-LABEL: @test_mm_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.vpermi2var.d.128
  return _mm_permutex2var_epi32(__A,__I,__B); 
}
__m128i test_mm_mask_permutex2var_epi32(__m128i __A, __mmask8 __U, __m128i __I, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.vpermi2var.d.128
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_permutex2var_epi32(__A,__U,__I,__B); 
}
__m128i test_mm_maskz_permutex2var_epi32(__mmask8 __U, __m128i __A, __m128i __I,  __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.vpermi2var.d.128
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_permutex2var_epi32(__U,__A,__I,__B); 
}
__m256i test_mm256_permutex2var_epi32(__m256i __A, __m256i __I, __m256i __B) {
  // CHECK-LABEL: @test_mm256_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.vpermi2var.d.256
  return _mm256_permutex2var_epi32(__A,__I,__B); 
}
__m256i test_mm256_mask_permutex2var_epi32(__m256i __A, __mmask8 __U, __m256i __I, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.vpermi2var.d.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_permutex2var_epi32(__A,__U,__I,__B); 
}
__m256i test_mm256_maskz_permutex2var_epi32(__mmask8 __U, __m256i __A, __m256i __I, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.vpermi2var.d.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_permutex2var_epi32(__U,__A,__I,__B); 
}
__m128d test_mm_permutex2var_pd(__m128d __A, __m128i __I, __m128d __B) {
  // CHECK-LABEL: @test_mm_permutex2var_pd
  // CHECK: @llvm.x86.avx512.vpermi2var.pd.128
  return _mm_permutex2var_pd(__A,__I,__B); 
}
__m128d test_mm_mask_permutex2var_pd(__m128d __A, __mmask8 __U, __m128i __I, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_permutex2var_pd
  // CHECK: @llvm.x86.avx512.vpermi2var.pd.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_permutex2var_pd(__A,__U,__I,__B); 
}
__m128d test_mm_maskz_permutex2var_pd(__mmask8 __U, __m128d __A, __m128i __I, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_permutex2var_pd
  // CHECK: @llvm.x86.avx512.vpermi2var.pd.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_permutex2var_pd(__U,__A,__I,__B); 
}
__m256d test_mm256_permutex2var_pd(__m256d __A, __m256i __I, __m256d __B) {
  // CHECK-LABEL: @test_mm256_permutex2var_pd
  // CHECK: @llvm.x86.avx512.vpermi2var.pd.256
  return _mm256_permutex2var_pd(__A,__I,__B); 
}
__m256d test_mm256_mask_permutex2var_pd(__m256d __A, __mmask8 __U, __m256i __I, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_permutex2var_pd
  // CHECK: @llvm.x86.avx512.vpermi2var.pd.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_permutex2var_pd(__A,__U,__I,__B); 
}
__m256d test_mm256_maskz_permutex2var_pd(__mmask8 __U, __m256d __A, __m256i __I,  __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_permutex2var_pd
  // CHECK: @llvm.x86.avx512.vpermi2var.pd.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_permutex2var_pd(__U,__A,__I,__B); 
}
__m128 test_mm_permutex2var_ps(__m128 __A, __m128i __I, __m128 __B) {
  // CHECK-LABEL: @test_mm_permutex2var_ps
  // CHECK: @llvm.x86.avx512.vpermi2var.ps.128
  return _mm_permutex2var_ps(__A,__I,__B); 
}
__m128 test_mm_mask_permutex2var_ps(__m128 __A, __mmask8 __U, __m128i __I, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_permutex2var_ps
  // CHECK: @llvm.x86.avx512.vpermi2var.ps.128
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_permutex2var_ps(__A,__U,__I,__B); 
}
__m128 test_mm_maskz_permutex2var_ps(__mmask8 __U, __m128 __A, __m128i __I, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_permutex2var_ps
  // CHECK: @llvm.x86.avx512.vpermi2var.ps.128
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_permutex2var_ps(__U,__A,__I,__B); 
}
__m256 test_mm256_permutex2var_ps(__m256 __A, __m256i __I, __m256 __B) {
  // CHECK-LABEL: @test_mm256_permutex2var_ps
  // CHECK: @llvm.x86.avx512.vpermi2var.ps.256
  return _mm256_permutex2var_ps(__A,__I,__B); 
}
__m256 test_mm256_mask_permutex2var_ps(__m256 __A, __mmask8 __U, __m256i __I, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_permutex2var_ps
  // CHECK: @llvm.x86.avx512.vpermi2var.ps.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_permutex2var_ps(__A,__U,__I,__B); 
}
__m256 test_mm256_maskz_permutex2var_ps(__mmask8 __U, __m256 __A, __m256i __I, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_permutex2var_ps
  // CHECK: @llvm.x86.avx512.vpermi2var.ps.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_permutex2var_ps(__U,__A,__I,__B); 
}
__m128i test_mm_permutex2var_epi64(__m128i __A, __m128i __I, __m128i __B) {
  // CHECK-LABEL: @test_mm_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.vpermi2var.q.128
  return _mm_permutex2var_epi64(__A,__I,__B); 
}
__m128i test_mm_mask_permutex2var_epi64(__m128i __A, __mmask8 __U, __m128i __I, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.vpermi2var.q.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_permutex2var_epi64(__A,__U,__I,__B); 
}
__m128i test_mm_maskz_permutex2var_epi64(__mmask8 __U, __m128i __A, __m128i __I, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.vpermi2var.q.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_permutex2var_epi64(__U,__A,__I,__B); 
}
__m256i test_mm256_permutex2var_epi64(__m256i __A, __m256i __I, __m256i __B) {
  // CHECK-LABEL: @test_mm256_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.vpermi2var.q.256
  return _mm256_permutex2var_epi64(__A,__I,__B); 
}
__m256i test_mm256_mask_permutex2var_epi64(__m256i __A, __mmask8 __U, __m256i __I, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.vpermi2var.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_permutex2var_epi64(__A,__U,__I,__B); 
}
__m256i test_mm256_maskz_permutex2var_epi64(__mmask8 __U, __m256i __A, __m256i __I, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.vpermi2var.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_permutex2var_epi64(__U,__A,__I,__B); 
}

__m128i test_mm_mask_cvtepi8_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi8_epi32
  // CHECK: sext <4 x i8> %{{.*}} to <4 x i32>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_cvtepi8_epi32(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepi8_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi8_epi32
  // CHECK: sext <4 x i8> %{{.*}} to <4 x i32>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_cvtepi8_epi32(__U, __A); 
}

__m256i test_mm256_mask_cvtepi8_epi32(__m256i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi8_epi32
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i32>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_cvtepi8_epi32(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepi8_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi8_epi32
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i32>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_cvtepi8_epi32(__U, __A); 
}

__m128i test_mm_mask_cvtepi8_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi8_epi64
  // CHECK: sext <2 x i8> %{{.*}} to <2 x i64>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_cvtepi8_epi64(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepi8_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi8_epi64
  // CHECK: sext <2 x i8> %{{.*}} to <2 x i64>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_cvtepi8_epi64(__U, __A); 
}

__m256i test_mm256_mask_cvtepi8_epi64(__m256i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi8_epi64
  // CHECK: sext <4 x i8> %{{.*}} to <4 x i64>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_cvtepi8_epi64(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepi8_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi8_epi64
  // CHECK: sext <4 x i8> %{{.*}} to <4 x i64>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_cvtepi8_epi64(__U, __A); 
}

__m128i test_mm_mask_cvtepi32_epi64(__m128i __W, __mmask8 __U, __m128i __X) {
  // CHECK-LABEL: @test_mm_mask_cvtepi32_epi64
  // CHECK: sext <2 x i32> %{{.*}} to <2 x i64>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_cvtepi32_epi64(__W, __U, __X); 
}

__m128i test_mm_maskz_cvtepi32_epi64(__mmask8 __U, __m128i __X) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi32_epi64
  // CHECK: sext <2 x i32> %{{.*}} to <2 x i64>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_cvtepi32_epi64(__U, __X); 
}

__m256i test_mm256_mask_cvtepi32_epi64(__m256i __W, __mmask8 __U, __m128i __X) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi32_epi64
  // CHECK: sext <4 x i32> %{{.*}} to <4 x i64>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_cvtepi32_epi64(__W, __U, __X); 
}

__m256i test_mm256_maskz_cvtepi32_epi64(__mmask8 __U, __m128i __X) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi32_epi64
  // CHECK: sext <4 x i32> %{{.*}} to <4 x i64>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_cvtepi32_epi64(__U, __X); 
}

__m128i test_mm_mask_cvtepi16_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi16_epi32
  // CHECK: sext <4 x i16> %{{.*}} to <4 x i32>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_cvtepi16_epi32(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepi16_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi16_epi32
  // CHECK: sext <4 x i16> %{{.*}} to <4 x i32>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_cvtepi16_epi32(__U, __A); 
}

__m256i test_mm256_mask_cvtepi16_epi32(__m256i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi16_epi32
  // CHECK: sext <8 x i16> %{{.*}} to <8 x i32>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_cvtepi16_epi32(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepi16_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi16_epi32
  // CHECK: sext <8 x i16> %{{.*}} to <8 x i32>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_cvtepi16_epi32(__U, __A); 
}

__m128i test_mm_mask_cvtepi16_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi16_epi64
  // CHECK: sext <2 x i16> %{{.*}} to <2 x i64>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_cvtepi16_epi64(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepi16_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi16_epi64
  // CHECK: sext <2 x i16> %{{.*}} to <2 x i64>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_cvtepi16_epi64(__U, __A); 
}

__m256i test_mm256_mask_cvtepi16_epi64(__m256i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi16_epi64
  // CHECK: sext <4 x i16> %{{.*}} to <4 x i64>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_cvtepi16_epi64(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepi16_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi16_epi64
  // CHECK: sext <4 x i16> %{{.*}} to <4 x i64>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_cvtepi16_epi64(__U, __A); 
}

__m128i test_mm_mask_cvtepu8_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepu8_epi32
  // CHECK: zext <4 x i8> %{{.*}} to <4 x i32>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_cvtepu8_epi32(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepu8_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu8_epi32
  // CHECK: zext <4 x i8> %{{.*}} to <4 x i32>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_cvtepu8_epi32(__U, __A);
}

__m256i test_mm256_mask_cvtepu8_epi32(__m256i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu8_epi32
  // CHECK: zext <8 x i8> %{{.*}} to <8 x i32>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_cvtepu8_epi32(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepu8_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu8_epi32
  // CHECK: zext <8 x i8> %{{.*}} to <8 x i32>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_cvtepu8_epi32(__U, __A); 
}

__m128i test_mm_mask_cvtepu8_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepu8_epi64
  // CHECK: zext <2 x i8> %{{.*}} to <2 x i64>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_cvtepu8_epi64(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepu8_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu8_epi64
  // CHECK: zext <2 x i8> %{{.*}} to <2 x i64>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_cvtepu8_epi64(__U, __A); 
}

__m256i test_mm256_mask_cvtepu8_epi64(__m256i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu8_epi64
  // CHECK: zext <4 x i8> %{{.*}} to <4 x i64>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_cvtepu8_epi64(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepu8_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu8_epi64
  // CHECK: zext <4 x i8> %{{.*}} to <4 x i64>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_cvtepu8_epi64(__U, __A); 
}

__m128i test_mm_mask_cvtepu32_epi64(__m128i __W, __mmask8 __U, __m128i __X) {
  // CHECK-LABEL: @test_mm_mask_cvtepu32_epi64
  // CHECK: zext <2 x i32> %{{.*}} to <2 x i64>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_cvtepu32_epi64(__W, __U, __X); 
}

__m128i test_mm_maskz_cvtepu32_epi64(__mmask8 __U, __m128i __X) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu32_epi64
  // CHECK: zext <2 x i32> %{{.*}} to <2 x i64>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_cvtepu32_epi64(__U, __X); 
}

__m256i test_mm256_mask_cvtepu32_epi64(__m256i __W, __mmask8 __U, __m128i __X) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu32_epi64
  // CHECK: zext <4 x i32> %{{.*}} to <4 x i64>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_cvtepu32_epi64(__W, __U, __X); 
}

__m256i test_mm256_maskz_cvtepu32_epi64(__mmask8 __U, __m128i __X) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu32_epi64
  // CHECK: zext <4 x i32> %{{.*}} to <4 x i64>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_cvtepu32_epi64(__U, __X); 
}

__m128i test_mm_mask_cvtepu16_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepu16_epi32
  // CHECK: zext <4 x i16> %{{.*}} to <4 x i32>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_cvtepu16_epi32(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepu16_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu16_epi32
  // CHECK: zext <4 x i16> %{{.*}} to <4 x i32>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_cvtepu16_epi32(__U, __A); 
}

__m256i test_mm256_mask_cvtepu16_epi32(__m256i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu16_epi32
  // CHECK: zext <8 x i16> %{{.*}} to <8 x i32>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_cvtepu16_epi32(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepu16_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu16_epi32
  // CHECK: zext <8 x i16> %{{.*}} to <8 x i32>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_cvtepu16_epi32(__U, __A); 
}

__m128i test_mm_mask_cvtepu16_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepu16_epi64
  // CHECK: zext <2 x i16> %{{.*}} to <2 x i64>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_cvtepu16_epi64(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepu16_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu16_epi64
  // CHECK: zext <2 x i16> %{{.*}} to <2 x i64>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_cvtepu16_epi64(__U, __A); 
}

__m256i test_mm256_mask_cvtepu16_epi64(__m256i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu16_epi64
  // CHECK: zext <4 x i16> %{{.*}} to <4 x i64>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_cvtepu16_epi64(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepu16_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu16_epi64
  // CHECK: zext <4 x i16> %{{.*}} to <4 x i64>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_cvtepu16_epi64(__U, __A); 
}

__m128i test_mm_rol_epi32(__m128i __A) {
  // CHECK-LABEL: @test_mm_rol_epi32
  // CHECK: @llvm.fshl.v4i32
  return _mm_rol_epi32(__A, 5); 
}

__m128i test_mm_mask_rol_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_rol_epi32
  // CHECK: @llvm.fshl.v4i32
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_rol_epi32(__W, __U, __A, 5); 
}

__m128i test_mm_maskz_rol_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_rol_epi32
  // CHECK: @llvm.fshl.v4i32
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_rol_epi32(__U, __A, 5); 
}

__m256i test_mm256_rol_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm256_rol_epi32
  // CHECK: @llvm.fshl.v8i32
  return _mm256_rol_epi32(__A, 5); 
}

__m256i test_mm256_mask_rol_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_rol_epi32
  // CHECK: @llvm.fshl.v8i32
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_rol_epi32(__W, __U, __A, 5); 
}

__m256i test_mm256_maskz_rol_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_rol_epi32
  // CHECK: @llvm.fshl.v8i32
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_rol_epi32(__U, __A, 5); 
}

__m128i test_mm_rol_epi64(__m128i __A) {
  // CHECK-LABEL: @test_mm_rol_epi64
  // CHECK: @llvm.fshl.v2i64
  return _mm_rol_epi64(__A, 5); 
}

__m128i test_mm_mask_rol_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_rol_epi64
  // CHECK: @llvm.fshl.v2i64
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_rol_epi64(__W, __U, __A, 5); 
}

__m128i test_mm_maskz_rol_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_rol_epi64
  // CHECK: @llvm.fshl.v2i64
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_rol_epi64(__U, __A, 5); 
}

__m256i test_mm256_rol_epi64(__m256i __A) {
  // CHECK-LABEL: @test_mm256_rol_epi64
  // CHECK: @llvm.fshl.v4i64
  return _mm256_rol_epi64(__A, 5); 
}

__m256i test_mm256_mask_rol_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_rol_epi64
  // CHECK: @llvm.fshl.v4i64
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_rol_epi64(__W, __U, __A, 5); 
}

__m256i test_mm256_maskz_rol_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_rol_epi64
  // CHECK: @llvm.fshl.v4i64
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_rol_epi64(__U, __A, 5); 
}

__m128i test_mm_rolv_epi32(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_rolv_epi32
  // CHECK: llvm.fshl.v4i32
  return _mm_rolv_epi32(__A, __B); 
}

__m128i test_mm_mask_rolv_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_rolv_epi32
  // CHECK: llvm.fshl.v4i32
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_rolv_epi32(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_rolv_epi32(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_rolv_epi32
  // CHECK: llvm.fshl.v4i32
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_rolv_epi32(__U, __A, __B); 
}

__m256i test_mm256_rolv_epi32(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_rolv_epi32
  // CHECK: @llvm.fshl.v8i32
  return _mm256_rolv_epi32(__A, __B); 
}

__m256i test_mm256_mask_rolv_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_rolv_epi32
  // CHECK: @llvm.fshl.v8i32
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_rolv_epi32(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_rolv_epi32(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_rolv_epi32
  // CHECK: @llvm.fshl.v8i32
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_rolv_epi32(__U, __A, __B); 
}

__m128i test_mm_rolv_epi64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_rolv_epi64
  // CHECK: @llvm.fshl.v2i64
  return _mm_rolv_epi64(__A, __B); 
}

__m128i test_mm_mask_rolv_epi64(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_rolv_epi64
  // CHECK: @llvm.fshl.v2i64
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_rolv_epi64(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_rolv_epi64(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_rolv_epi64
  // CHECK: @llvm.fshl.v2i64
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_rolv_epi64(__U, __A, __B); 
}

__m256i test_mm256_rolv_epi64(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_rolv_epi64
  // CHECK: @llvm.fshl.v4i64
  return _mm256_rolv_epi64(__A, __B); 
}

__m256i test_mm256_mask_rolv_epi64(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_rolv_epi64
  // CHECK: @llvm.fshl.v4i64
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_rolv_epi64(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_rolv_epi64(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_rolv_epi64
  // CHECK: @llvm.fshl.v4i64
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_rolv_epi64(__U, __A, __B); 
}

__m128i test_mm_ror_epi32(__m128i __A) {
  // CHECK-LABEL: @test_mm_ror_epi32
  // CHECK: @llvm.fshr.v4i32
  return _mm_ror_epi32(__A, 5); 
}

__m128i test_mm_mask_ror_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_ror_epi32
  // CHECK: @llvm.fshr.v4i32
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_ror_epi32(__W, __U, __A, 5); 
}

__m128i test_mm_maskz_ror_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_ror_epi32
  // CHECK: @llvm.fshr.v4i32
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_ror_epi32(__U, __A, 5); 
}

__m256i test_mm256_ror_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm256_ror_epi32
  // CHECK: @llvm.fshr.v8i32
  return _mm256_ror_epi32(__A, 5); 
}

__m256i test_mm256_mask_ror_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_ror_epi32
  // CHECK: @llvm.fshr.v8i32
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_ror_epi32(__W, __U, __A, 5); 
}

__m256i test_mm256_maskz_ror_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_ror_epi32
  // CHECK: @llvm.fshr.v8i32
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_ror_epi32(__U, __A, 5); 
}

__m128i test_mm_ror_epi64(__m128i __A) {
  // CHECK-LABEL: @test_mm_ror_epi64
  // CHECK: @llvm.fshr.v2i64
  return _mm_ror_epi64(__A, 5); 
}

__m128i test_mm_mask_ror_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_ror_epi64
  // CHECK: @llvm.fshr.v2i64
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_ror_epi64(__W, __U, __A, 5); 
}

__m128i test_mm_maskz_ror_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_ror_epi64
  // CHECK: @llvm.fshr.v2i64
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_ror_epi64(__U, __A, 5); 
}

__m256i test_mm256_ror_epi64(__m256i __A) {
  // CHECK-LABEL: @test_mm256_ror_epi64
  // CHECK: @llvm.fshr.v4i64
  return _mm256_ror_epi64(__A, 5); 
}

__m256i test_mm256_mask_ror_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_ror_epi64
  // CHECK: @llvm.fshr.v4i64
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_ror_epi64(__W, __U, __A,5); 
}

__m256i test_mm256_maskz_ror_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_ror_epi64
  // CHECK: @llvm.fshr.v4i64
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_ror_epi64(__U, __A, 5); 
}


__m128i test_mm_rorv_epi32(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_rorv_epi32
  // CHECK: @llvm.fshr.v4i32
  return _mm_rorv_epi32(__A, __B); 
}

__m128i test_mm_mask_rorv_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_rorv_epi32
  // CHECK: @llvm.fshr.v4i32
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_rorv_epi32(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_rorv_epi32(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_rorv_epi32
  // CHECK: @llvm.fshr.v4i32
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_rorv_epi32(__U, __A, __B); 
}

__m256i test_mm256_rorv_epi32(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_rorv_epi32
  // CHECK: @llvm.fshr.v8i32
  return _mm256_rorv_epi32(__A, __B); 
}

__m256i test_mm256_mask_rorv_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_rorv_epi32
  // CHECK: @llvm.fshr.v8i32
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_rorv_epi32(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_rorv_epi32(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_rorv_epi32
  // CHECK: @llvm.fshr.v8i32
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_rorv_epi32(__U, __A, __B); 
}

__m128i test_mm_rorv_epi64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_rorv_epi64
  // CHECK: @llvm.fshr.v2i64
  return _mm_rorv_epi64(__A, __B); 
}

__m128i test_mm_mask_rorv_epi64(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_rorv_epi64
  // CHECK: @llvm.fshr.v2i64
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_rorv_epi64(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_rorv_epi64(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_rorv_epi64
  // CHECK: @llvm.fshr.v2i64
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_rorv_epi64(__U, __A, __B); 
}

__m256i test_mm256_rorv_epi64(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_rorv_epi64
  // CHECK: @llvm.fshr.v4i64
  return _mm256_rorv_epi64(__A, __B); 
}

__m256i test_mm256_mask_rorv_epi64(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_rorv_epi64
  // CHECK: @llvm.fshr.v4i64
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_rorv_epi64(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_rorv_epi64(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_rorv_epi64
  // CHECK: @llvm.fshr.v4i64
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_rorv_epi64(__U, __A, __B); 
}

__m128i test_mm_mask_sllv_epi64(__m128i __W, __mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_mask_sllv_epi64
  // CHECK: @llvm.x86.avx2.psllv.q
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_sllv_epi64(__W, __U, __X, __Y); 
}

__m128i test_mm_maskz_sllv_epi64(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_maskz_sllv_epi64
  // CHECK: @llvm.x86.avx2.psllv.q
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_sllv_epi64(__U, __X, __Y); 
}

__m256i test_mm256_mask_sllv_epi64(__m256i __W, __mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_sllv_epi64
  // CHECK: @llvm.x86.avx2.psllv.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_sllv_epi64(__W, __U, __X, __Y); 
}

__m256i test_mm256_maskz_sllv_epi64(__mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_sllv_epi64
  // CHECK: @llvm.x86.avx2.psllv.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_sllv_epi64(__U, __X, __Y); 
}

__m128i test_mm_mask_sllv_epi32(__m128i __W, __mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_mask_sllv_epi32
  // CHECK: @llvm.x86.avx2.psllv.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_sllv_epi32(__W, __U, __X, __Y); 
}

__m128i test_mm_maskz_sllv_epi32(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_maskz_sllv_epi32
  // CHECK: @llvm.x86.avx2.psllv.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_sllv_epi32(__U, __X, __Y); 
}

__m256i test_mm256_mask_sllv_epi32(__m256i __W, __mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_sllv_epi32
  // CHECK: @llvm.x86.avx2.psllv.d.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_sllv_epi32(__W, __U, __X, __Y); 
}

__m256i test_mm256_maskz_sllv_epi32(__mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_sllv_epi32
  // CHECK: @llvm.x86.avx2.psllv.d.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_sllv_epi32(__U, __X, __Y); 
}

__m128i test_mm_mask_srlv_epi64(__m128i __W, __mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_mask_srlv_epi64
  // CHECK: @llvm.x86.avx2.psrlv.q
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_srlv_epi64(__W, __U, __X, __Y); 
}

__m128i test_mm_maskz_srlv_epi64(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_maskz_srlv_epi64
  // CHECK: @llvm.x86.avx2.psrlv.q
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_srlv_epi64(__U, __X, __Y); 
}

__m256i test_mm256_mask_srlv_epi64(__m256i __W, __mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_srlv_epi64
  // CHECK: @llvm.x86.avx2.psrlv.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_srlv_epi64(__W, __U, __X, __Y); 
}

__m256i test_mm256_maskz_srlv_epi64(__mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_srlv_epi64
  // CHECK: @llvm.x86.avx2.psrlv.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_srlv_epi64(__U, __X, __Y); 
}

__m128i test_mm_mask_srlv_epi32(__m128i __W, __mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_mask_srlv_epi32
  // CHECK: @llvm.x86.avx2.psrlv.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_srlv_epi32(__W, __U, __X, __Y); 
}

__m128i test_mm_maskz_srlv_epi32(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_maskz_srlv_epi32
  // CHECK: @llvm.x86.avx2.psrlv.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_srlv_epi32(__U, __X, __Y); 
}

__m256i test_mm256_mask_srlv_epi32(__m256i __W, __mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_srlv_epi32
  // CHECK: @llvm.x86.avx2.psrlv.d.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_srlv_epi32(__W, __U, __X, __Y); 
}

__m256i test_mm256_maskz_srlv_epi32(__mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_srlv_epi32
  // CHECK: @llvm.x86.avx2.psrlv.d.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_srlv_epi32(__U, __X, __Y); 
}

__m128i test_mm_mask_srl_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_srl_epi32
  // CHECK: @llvm.x86.sse2.psrl.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_srl_epi32(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_srl_epi32(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_srl_epi32
  // CHECK: @llvm.x86.sse2.psrl.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_srl_epi32(__U, __A, __B); 
}

__m256i test_mm256_mask_srl_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_mask_srl_epi32
  // CHECK: @llvm.x86.avx2.psrl.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_srl_epi32(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_srl_epi32(__mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_srl_epi32
  // CHECK: @llvm.x86.avx2.psrl.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_srl_epi32(__U, __A, __B); 
}

__m128i test_mm_mask_srli_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_srli_epi32
  // CHECK: @llvm.x86.sse2.psrli.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_srli_epi32(__W, __U, __A, 5); 
}

__m128i test_mm_mask_srli_epi32_2(__m128i __W, __mmask8 __U, __m128i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm_mask_srli_epi32_2
  // CHECK: @llvm.x86.sse2.psrli.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_srli_epi32(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_srli_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_srli_epi32
  // CHECK: @llvm.x86.sse2.psrli.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_srli_epi32(__U, __A, 5); 
}

__m128i test_mm_maskz_srli_epi32_2(__mmask8 __U, __m128i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm_maskz_srli_epi32_2
  // CHECK: @llvm.x86.sse2.psrli.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_srli_epi32(__U, __A, __B); 
}

__m256i test_mm256_mask_srli_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_srli_epi32
  // CHECK: @llvm.x86.avx2.psrli.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_srli_epi32(__W, __U, __A, 5); 
}

__m256i test_mm256_mask_srli_epi32_2(__m256i __W, __mmask8 __U, __m256i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm256_mask_srli_epi32_2
  // CHECK: @llvm.x86.avx2.psrli.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_srli_epi32(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_srli_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_srli_epi32
  // CHECK: @llvm.x86.avx2.psrli.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_srli_epi32(__U, __A, 5); 
}

__m256i test_mm256_maskz_srli_epi32_2(__mmask8 __U, __m256i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm256_maskz_srli_epi32_2
  // CHECK: @llvm.x86.avx2.psrli.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_srli_epi32(__U, __A, __B); 
}
__m128i test_mm_mask_srl_epi64(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_srl_epi64
  // CHECK: @llvm.x86.sse2.psrl.q
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_srl_epi64(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_srl_epi64(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_srl_epi64
  // CHECK: @llvm.x86.sse2.psrl.q
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_srl_epi64(__U, __A, __B); 
}

__m256i test_mm256_mask_srl_epi64(__m256i __W, __mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_mask_srl_epi64
  // CHECK: @llvm.x86.avx2.psrl.q
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_srl_epi64(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_srl_epi64(__mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_srl_epi64
  // CHECK: @llvm.x86.avx2.psrl.q
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_srl_epi64(__U, __A, __B); 
}

__m128i test_mm_mask_srli_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_srli_epi64
  // CHECK: @llvm.x86.sse2.psrli.q
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_srli_epi64(__W, __U, __A, 5); 
}

__m128i test_mm_mask_srli_epi64_2(__m128i __W, __mmask8 __U, __m128i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm_mask_srli_epi64_2
  // CHECK: @llvm.x86.sse2.psrli.q
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_srli_epi64(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_srli_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_srli_epi64
  // CHECK: @llvm.x86.sse2.psrli.q
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_srli_epi64(__U, __A, 5); 
}

__m128i test_mm_maskz_srli_epi64_2(__mmask8 __U, __m128i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm_maskz_srli_epi64_2
  // CHECK: @llvm.x86.sse2.psrli.q
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_srli_epi64(__U, __A, __B); 
}

__m256i test_mm256_mask_srli_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_srli_epi64
  // CHECK: @llvm.x86.avx2.psrli.q
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_srli_epi64(__W, __U, __A, 5); 
}

__m256i test_mm256_mask_srli_epi64_2(__m256i __W, __mmask8 __U, __m256i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm256_mask_srli_epi64_2
  // CHECK: @llvm.x86.avx2.psrli.q
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_srli_epi64(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_srli_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_srli_epi64
  // CHECK: @llvm.x86.avx2.psrli.q
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_srli_epi64(__U, __A, 5); 
}

__m256i test_mm256_maskz_srli_epi64_2(__mmask8 __U,__m256i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm256_maskz_srli_epi64_2
  // CHECK: @llvm.x86.avx2.psrli.q
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_srli_epi64(__U, __A, __B); 
}

__m128i test_mm_mask_sll_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_sll_epi32
  // CHECK: @llvm.x86.sse2.psll.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_sll_epi32(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_sll_epi32(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_sll_epi32
  // CHECK: @llvm.x86.sse2.psll.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_sll_epi32(__U, __A, __B); 
}

__m256i test_mm256_mask_sll_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_mask_sll_epi32
  // CHECK: @llvm.x86.avx2.psll.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_sll_epi32(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_sll_epi32(__mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_sll_epi32
  // CHECK: @llvm.x86.avx2.psll.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_sll_epi32(__U, __A, __B); 
}

__m128i test_mm_mask_slli_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_slli_epi32
  // CHECK: @llvm.x86.sse2.pslli.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_slli_epi32(__W, __U, __A, 5); 
}

__m128i test_mm_mask_slli_epi32_2(__m128i __W, __mmask8 __U, __m128i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm_mask_slli_epi32_2
  // CHECK: @llvm.x86.sse2.pslli.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_slli_epi32(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_slli_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_slli_epi32
  // CHECK: @llvm.x86.sse2.pslli.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_slli_epi32(__U, __A, 5); 
}

__m128i test_mm_maskz_slli_epi32_2(__mmask8 __U, __m128i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm_maskz_slli_epi32_2
  // CHECK: @llvm.x86.sse2.pslli.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_slli_epi32(__U, __A, __B); 
}

__m256i test_mm256_mask_slli_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_slli_epi32
  // CHECK: @llvm.x86.avx2.pslli.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_slli_epi32(__W, __U, __A, 5); 
}

__m256i test_mm256_mask_slli_epi32_2(__m256i __W, __mmask8 __U, __m256i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm256_mask_slli_epi32_2
  // CHECK: @llvm.x86.avx2.pslli.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_slli_epi32(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_slli_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_slli_epi32
  // CHECK: @llvm.x86.avx2.pslli.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_slli_epi32(__U, __A, 5); 
}

__m256i test_mm256_maskz_slli_epi32_2(__mmask8 __U, __m256i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm256_maskz_slli_epi32_2
  // CHECK: @llvm.x86.avx2.pslli.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_slli_epi32(__U, __A, __B); 
}

__m128i test_mm_mask_sll_epi64(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_sll_epi64
  // CHECK: @llvm.x86.sse2.psll.q
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_sll_epi64(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_sll_epi64(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_sll_epi64
  // CHECK: @llvm.x86.sse2.psll.q
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_sll_epi64(__U, __A, __B); 
}

__m256i test_mm256_mask_sll_epi64(__m256i __W, __mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_mask_sll_epi64
  // CHECK: @llvm.x86.avx2.psll.q
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_sll_epi64(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_sll_epi64(__mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_sll_epi64
  // CHECK: @llvm.x86.avx2.psll.q
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_sll_epi64(__U, __A, __B); 
}

__m128i test_mm_mask_slli_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_slli_epi64
  // CHECK: @llvm.x86.sse2.pslli.q
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_slli_epi64(__W, __U, __A, 5); 
}

__m128i test_mm_mask_slli_epi64_2(__m128i __W, __mmask8 __U, __m128i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm_mask_slli_epi64_2
  // CHECK: @llvm.x86.sse2.pslli.q
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_slli_epi64(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_slli_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_slli_epi64
  // CHECK: @llvm.x86.sse2.pslli.q
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_slli_epi64(__U, __A, 5); 
}

__m128i test_mm_maskz_slli_epi64_2(__mmask8 __U, __m128i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm_maskz_slli_epi64_2
  // CHECK: @llvm.x86.sse2.pslli.q
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_slli_epi64(__U, __A, __B); 
}

__m256i test_mm256_mask_slli_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_slli_epi64
  // CHECK: @llvm.x86.avx2.pslli.q
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_slli_epi64(__W, __U, __A, 5); 
}

__m256i test_mm256_mask_slli_epi64_2(__m256i __W, __mmask8 __U, __m256i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm256_mask_slli_epi64_2
  // CHECK: @llvm.x86.avx2.pslli.q
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_slli_epi64(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_slli_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_slli_epi64
  // CHECK: @llvm.x86.avx2.pslli.q
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_slli_epi64(__U, __A, 5); 
}

__m256i test_mm256_maskz_slli_epi64_2(__mmask8 __U, __m256i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm256_maskz_slli_epi64_2
  // CHECK: @llvm.x86.avx2.pslli.q
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_slli_epi64(__U, __A, __B);
}

__m128i test_mm_mask_srav_epi32(__m128i __W, __mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_mask_srav_epi32
  // CHECK: @llvm.x86.avx2.psrav.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_srav_epi32(__W, __U, __X, __Y); 
}

__m128i test_mm_maskz_srav_epi32(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_maskz_srav_epi32
  // CHECK: @llvm.x86.avx2.psrav.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_srav_epi32(__U, __X, __Y); 
}

__m256i test_mm256_mask_srav_epi32(__m256i __W, __mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_srav_epi32
  // CHECK: @llvm.x86.avx2.psrav.d.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_srav_epi32(__W, __U, __X, __Y); 
}

__m256i test_mm256_maskz_srav_epi32(__mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_srav_epi32
  // CHECK: @llvm.x86.avx2.psrav.d.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_srav_epi32(__U, __X, __Y); 
}

__m128i test_mm_srav_epi64(__m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_srav_epi64
  // CHECK: @llvm.x86.avx512.psrav.q.128
  return _mm_srav_epi64(__X, __Y); 
}

__m128i test_mm_mask_srav_epi64(__m128i __W, __mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_mask_srav_epi64
  // CHECK: @llvm.x86.avx512.psrav.q.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_srav_epi64(__W, __U, __X, __Y); 
}

__m128i test_mm_maskz_srav_epi64(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_maskz_srav_epi64
  // CHECK: @llvm.x86.avx512.psrav.q.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_srav_epi64(__U, __X, __Y); 
}

__m256i test_mm256_srav_epi64(__m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_srav_epi64
  // CHECK: @llvm.x86.avx512.psrav.q.256
  return _mm256_srav_epi64(__X, __Y); 
}

__m256i test_mm256_mask_srav_epi64(__m256i __W, __mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_srav_epi64
  // CHECK: @llvm.x86.avx512.psrav.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_srav_epi64(__W, __U, __X, __Y); 
}

__m256i test_mm256_maskz_srav_epi64(__mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_srav_epi64
  // CHECK: @llvm.x86.avx512.psrav.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_srav_epi64(__U, __X, __Y); 
}

void test_mm_store_epi32(void *__P, __m128i __A) {
  // CHECK-LABEL: @test_mm_store_epi32
  // CHECK: store <2 x i64> %{{.*}}, <2 x i64>* %{{.*}}
  return _mm_store_epi32(__P, __A);
}

void test_mm_mask_store_epi32(void *__P, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_store_epi32
  // CHECK: @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %{{.*}}, <4 x i32>* %{{.}}, i32 16, <4 x i1> %{{.*}})
  return _mm_mask_store_epi32(__P, __U, __A); 
}

void test_mm256_store_epi32(void *__P, __m256i __A) {
  // CHECK-LABEL: @test_mm256_store_epi32
  // CHECK: store <4 x i64> %{{.*}}, <4 x i64>* %{{.*}}
  return _mm256_store_epi32(__P, __A);
}

void test_mm256_mask_store_epi32(void *__P, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_store_epi32
  // CHECK: @llvm.masked.store.v8i32.p0v8i32(<8 x i32> %{{.*}}, <8 x i32>* %{{.}}, i32 32, <8 x i1> %{{.*}})
  return _mm256_mask_store_epi32(__P, __U, __A); 
}

__m128i test_mm_mask_mov_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_mov_epi32
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_mov_epi32(__W, __U, __A); 
}

__m128i test_mm_maskz_mov_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_mov_epi32
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_mov_epi32(__U, __A); 
}

__m256i test_mm256_mask_mov_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_mov_epi32
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_mov_epi32(__W, __U, __A); 
}

__m256i test_mm256_maskz_mov_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_mov_epi32
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_mov_epi32(__U, __A); 
}

__m128i test_mm_mask_mov_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_mov_epi64
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_mov_epi64(__W, __U, __A); 
}

__m128i test_mm_maskz_mov_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_mov_epi64
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_mov_epi64(__U, __A); 
}

__m256i test_mm256_mask_mov_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_mov_epi64
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_mov_epi64(__W, __U, __A); 
}

__m256i test_mm256_maskz_mov_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_mov_epi64
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_mov_epi64(__U, __A); 
}

__m128i test_mm_load_epi32(void const *__P) {
  // CHECK-LABEL: @test_mm_load_epi32
  // CHECK: load <2 x i64>, <2 x i64>* %{{.*}}
  return _mm_load_epi32(__P);
}

__m128i test_mm_mask_load_epi32(__m128i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_load_epi32
  // CHECK: @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %{{.*}}, i32 16, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_mask_load_epi32(__W, __U, __P); 
}

__m128i test_mm_maskz_load_epi32(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_load_epi32
  // CHECK: @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %{{.*}}, i32 16, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maskz_load_epi32(__U, __P); 
}

__m256i test_mm256_load_epi32(void const *__P) {
  // CHECK-LABEL: @test_mm256_load_epi32
  // CHECK: load <4 x i64>, <4 x i64>* %{{.*}}
  return _mm256_load_epi32(__P);
}

__m256i test_mm256_mask_load_epi32(__m256i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_load_epi32
  // CHECK: @llvm.masked.load.v8i32.p0v8i32(<8 x i32>* %{{.*}}, i32 32, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_mask_load_epi32(__W, __U, __P); 
}

__m256i test_mm256_maskz_load_epi32(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_load_epi32
  // CHECK: @llvm.masked.load.v8i32.p0v8i32(<8 x i32>* %{{.*}}, i32 32, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_maskz_load_epi32(__U, __P); 
}

__m128i test_mm_load_epi64(void const *__P) {
  // CHECK-LABEL: @test_mm_load_epi64
  // CHECK: load <2 x i64>, <2 x i64>* %{{.*}}
  return _mm_load_epi64(__P);
}

__m128i test_mm_mask_load_epi64(__m128i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_load_epi64
  // CHECK: @llvm.masked.load.v2i64.p0v2i64(<2 x i64>* %{{.*}}, i32 16, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_mask_load_epi64(__W, __U, __P); 
}

__m128i test_mm_maskz_load_epi64(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_load_epi64
  // CHECK: @llvm.masked.load.v2i64.p0v2i64(<2 x i64>* %{{.*}}, i32 16, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_maskz_load_epi64(__U, __P); 
}

__m256i test_mm256_load_epi64(void const *__P) {
  // CHECK-LABEL: @test_mm256_load_epi64
  // CHECK: load <4 x i64>, <4 x i64>* %{{.*}}
  return _mm256_load_epi64(__P);
}

__m256i test_mm256_mask_load_epi64(__m256i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_load_epi64
  // CHECK: @llvm.masked.load.v4i64.p0v4i64(<4 x i64>* %{{.*}}, i32 32, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_mask_load_epi64(__W, __U, __P); 
}

__m256i test_mm256_maskz_load_epi64(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_load_epi64
  // CHECK: @llvm.masked.load.v4i64.p0v4i64(<4 x i64>* %{{.*}}, i32 32, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_maskz_load_epi64(__U, __P); 
}

void test_mm_store_epi64(void *__P, __m128i __A) {
  // CHECK-LABEL: @test_mm_store_epi64
  // CHECK: store <2 x i64> %{{.*}}, <2 x i64>* %{{.*}}
  return _mm_store_epi64(__P, __A);
}

void test_mm_mask_store_epi64(void *__P, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_store_epi64
  // CHECK: @llvm.masked.store.v2i64.p0v2i64(<2 x i64> %{{.*}}, <2 x i64>* %{{.*}}, i32 16, <2 x i1> %{{.*}})
  return _mm_mask_store_epi64(__P, __U, __A); 
}

void test_mm256_store_epi64(void *__P, __m256i __A) {
  // CHECK-LABEL: @test_mm256_store_epi64
  // CHECK: store <4 x i64> %{{.*}}, <4 x i64>* %{{.*}}
  return _mm256_store_epi64(__P, __A);
}

void test_mm256_mask_store_epi64(void *__P, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_store_epi64
  // CHECK: @llvm.masked.store.v4i64.p0v4i64(<4 x i64> %{{.*}}, <4 x i64>* %{{.*}}, i32 32, <4 x i1> %{{.*}})
  return _mm256_mask_store_epi64(__P, __U, __A); 
}

__m128d test_mm_mask_movedup_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_movedup_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> zeroinitializer
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_movedup_pd(__W, __U, __A); 
}

__m128d test_mm_maskz_movedup_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_movedup_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> zeroinitializer
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_movedup_pd(__U, __A); 
}

__m256d test_mm256_mask_movedup_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_movedup_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_movedup_pd(__W, __U, __A); 
}

__m256d test_mm256_maskz_movedup_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_movedup_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_movedup_pd(__U, __A); 
}

__m128i test_mm_mask_set1_epi32(__m128i __O, __mmask8 __M) {
  // CHECK-LABEL: @test_mm_mask_set1_epi32
  // CHECK: insertelement <4 x i32> undef, i32 %{{.*}}, i32 0
  // CHECK: insertelement <4 x i32> %{{.*}}32 1
  // CHECK: insertelement <4 x i32> %{{.*}}32 2
  // CHECK: insertelement <4 x i32> %{{.*}}32 3
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}  
  return _mm_mask_set1_epi32(__O, __M, 5); 
}

__m128i test_mm_maskz_set1_epi32(__mmask8 __M) {
  // CHECK-LABEL: @test_mm_maskz_set1_epi32
  // CHECK: insertelement <4 x i32> undef, i32 %{{.*}}, i32 0
  // CHECK: insertelement <4 x i32> %{{.*}}32 1
  // CHECK: insertelement <4 x i32> %{{.*}}32 2
  // CHECK: insertelement <4 x i32> %{{.*}}32 3
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}  
  return _mm_maskz_set1_epi32(__M, 5); 
}

__m256i test_mm256_mask_set1_epi32(__m256i __O, __mmask8 __M) {
  // CHECK-LABEL: @test_mm256_mask_set1_epi32
  // CHECK:  insertelement <8 x i32> undef, i32 %{{.*}}, i32 0
  // CHECK:  insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 1
  // CHECK:  insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 2
  // CHECK:  insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 3
  // CHECK:  insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 4
  // CHECK:  insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 5
  // CHECK:  insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 6
  // CHECK:  insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 7
  // CHECK:  select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_set1_epi32(__O, __M, 5); 
}

__m256i test_mm256_maskz_set1_epi32(__mmask8 __M) {
  // CHECK-LABEL: @test_mm256_maskz_set1_epi32
  // CHECK:  insertelement <8 x i32> undef, i32 %{{.*}}, i32 0
  // CHECK:  insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 1
  // CHECK:  insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 2
  // CHECK:  insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 3
  // CHECK:  insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 4
  // CHECK:  insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 5
  // CHECK:  insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 6
  // CHECK:  insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 7
  // CHECK:  select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_set1_epi32(__M, 5); 
}

__m128i test_mm_mask_set1_epi64(__m128i __O, __mmask8 __M, long long __A) {
  // CHECK-LABEL: @test_mm_mask_set1_epi64
  // CHECK: insertelement <2 x i64> undef, i64 %{{.*}}, i32 0
  // CHECK: insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_set1_epi64(__O, __M, __A); 
}

__m128i test_mm_maskz_set1_epi64(__mmask8 __M, long long __A) {
  // CHECK-LABEL: @test_mm_maskz_set1_epi64
  // CHECK: insertelement <2 x i64> undef, i64 %{{.*}}, i32 0
  // CHECK: insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_set1_epi64(__M, __A); 
}

__m256i test_mm256_mask_set1_epi64(__m256i __O, __mmask8 __M, long long __A) {
  // CHECK-LABEL: @test_mm256_mask_set1_epi64
  // CHECK: insertelement <4 x i64> undef, i64 %{{.*}}, i32 0
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 2
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 3
  // CHECK:  shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK:  select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_set1_epi64(__O, __M, __A); 
}

__m256i test_mm256_maskz_set1_epi64(__mmask8 __M, long long __A) {
  // CHECK-LABEL: @test_mm256_maskz_set1_epi64
  // CHECK: insertelement <4 x i64> undef, i64 %{{.*}}, i32 0
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 2
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 3
  // CHECK:  shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK:  select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_set1_epi64(__M, __A); 
}

__m128d test_mm_fixupimm_pd(__m128d __A, __m128d __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_fixupimm_pd
  // CHECK: @llvm.x86.avx512.mask.fixupimm.pd.128
  return _mm_fixupimm_pd(__A, __B, __C, 5); 
}

__m128d test_mm_mask_fixupimm_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_fixupimm_pd
  // CHECK: @llvm.x86.avx512.mask.fixupimm.pd.128
  return _mm_mask_fixupimm_pd(__A, __U, __B, __C, 5); 
}

__m128d test_mm_maskz_fixupimm_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_maskz_fixupimm_pd
  // CHECK: @llvm.x86.avx512.maskz.fixupimm.pd.128
  return _mm_maskz_fixupimm_pd(__U, __A, __B, __C, 5); 
}

__m256d test_mm256_fixupimm_pd(__m256d __A, __m256d __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_fixupimm_pd
  // CHECK: @llvm.x86.avx512.mask.fixupimm.pd.256
  return _mm256_fixupimm_pd(__A, __B, __C, 5); 
}

__m256d test_mm256_mask_fixupimm_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_mask_fixupimm_pd
  // CHECK: @llvm.x86.avx512.mask.fixupimm.pd.256
  return _mm256_mask_fixupimm_pd(__A, __U, __B, __C, 5); 
}

__m256d test_mm256_maskz_fixupimm_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_maskz_fixupimm_pd
  // CHECK: @llvm.x86.avx512.maskz.fixupimm.pd.256
  return _mm256_maskz_fixupimm_pd(__U, __A, __B, __C, 5); 
}

__m128 test_mm_fixupimm_ps(__m128 __A, __m128 __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_fixupimm_ps
  // CHECK: @llvm.x86.avx512.mask.fixupimm.ps.128
  return _mm_fixupimm_ps(__A, __B, __C, 5); 
}

__m128 test_mm_mask_fixupimm_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_fixupimm_ps
  // CHECK: @llvm.x86.avx512.mask.fixupimm.ps.128
  return _mm_mask_fixupimm_ps(__A, __U, __B, __C, 5); 
}

__m128 test_mm_maskz_fixupimm_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_maskz_fixupimm_ps
  // CHECK: @llvm.x86.avx512.maskz.fixupimm.ps.128
  return _mm_maskz_fixupimm_ps(__U, __A, __B, __C, 5); 
}

__m256 test_mm256_fixupimm_ps(__m256 __A, __m256 __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_fixupimm_ps
  // CHECK: @llvm.x86.avx512.mask.fixupimm.ps.256
  return _mm256_fixupimm_ps(__A, __B, __C, 5); 
}

__m256 test_mm256_mask_fixupimm_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_mask_fixupimm_ps
  // CHECK: @llvm.x86.avx512.mask.fixupimm.ps.256
  return _mm256_mask_fixupimm_ps(__A, __U, __B, __C, 5); 
}

__m256 test_mm256_maskz_fixupimm_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_maskz_fixupimm_ps
  // CHECK: @llvm.x86.avx512.maskz.fixupimm.ps.256
  return _mm256_maskz_fixupimm_ps(__U, __A, __B, __C, 5); 
}

__m128d test_mm_mask_load_pd(__m128d __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_load_pd
  // CHECK: @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %{{.*}}, i32 16, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_mask_load_pd(__W, __U, __P); 
}

__m128d test_mm_maskz_load_pd(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_load_pd
  // CHECK: @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %{{.*}}, i32 16, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_maskz_load_pd(__U, __P); 
}

__m256d test_mm256_mask_load_pd(__m256d __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_load_pd
  // CHECK: @llvm.masked.load.v4f64.p0v4f64(<4 x double>* %{{.*}}, i32 32, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_mask_load_pd(__W, __U, __P); 
}

__m256d test_mm256_maskz_load_pd(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_load_pd
  // CHECK: @llvm.masked.load.v4f64.p0v4f64(<4 x double>* %{{.*}}, i32 32, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_maskz_load_pd(__U, __P); 
}

__m128 test_mm_mask_load_ps(__m128 __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_load_ps
  // CHECK: @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %{{.*}}, i32 16, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_mask_load_ps(__W, __U, __P); 
}

__m128 test_mm_maskz_load_ps(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_load_ps
  // CHECK: @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %{{.*}}, i32 16, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_maskz_load_ps(__U, __P); 
}

__m256 test_mm256_mask_load_ps(__m256 __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_load_ps
  // CHECK: @llvm.masked.load.v8f32.p0v8f32(<8 x float>* %{{.*}}, i32 32, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_mask_load_ps(__W, __U, __P); 
}

__m256 test_mm256_maskz_load_ps(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_load_ps
  // CHECK: @llvm.masked.load.v8f32.p0v8f32(<8 x float>* %{{.*}}, i32 32, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_maskz_load_ps(__U, __P); 
}

__m128i test_mm_loadu_epi64(void const *__P) {
  // CHECK-LABEL: @test_mm_loadu_epi64
  // CHECK: load <2 x i64>, <2 x i64>* %{{.*}}, align 1{{$}}
  return _mm_loadu_epi64(__P);
}

__m128i test_mm_mask_loadu_epi64(__m128i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_loadu_epi64
  // CHECK: @llvm.masked.load.v2i64.p0v2i64(<2 x i64>* %{{.*}}, i32 1, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_mask_loadu_epi64(__W, __U, __P); 
}

__m128i test_mm_maskz_loadu_epi64(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_loadu_epi64
  // CHECK: @llvm.masked.load.v2i64.p0v2i64(<2 x i64>* %{{.*}}, i32 1, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_maskz_loadu_epi64(__U, __P); 
}

__m256i test_mm256_loadu_epi64(void const *__P) {
  // CHECK-LABEL: @test_mm256_loadu_epi64
  // CHECK: load <4 x i64>, <4 x i64>* %{{.*}}, align 1{{$}}
  return _mm256_loadu_epi64(__P);
}

__m256i test_mm256_mask_loadu_epi64(__m256i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_loadu_epi64
  // CHECK: @llvm.masked.load.v4i64.p0v4i64(<4 x i64>* %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_mask_loadu_epi64(__W, __U, __P); 
}

__m256i test_mm256_maskz_loadu_epi64(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_loadu_epi64
  // CHECK: @llvm.masked.load.v4i64.p0v4i64(<4 x i64>* %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_maskz_loadu_epi64(__U, __P); 
}

__m128i test_mm_loadu_epi32(void const *__P) {
  // CHECK-LABEL: @test_mm_loadu_epi32
  // CHECK: load <2 x i64>, <2 x i64>* %{{.*}}, align 1{{$}}
  return _mm_loadu_epi32(__P);
}

__m128i test_mm_mask_loadu_epi32(__m128i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_loadu_epi32
  // CHECK: @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_mask_loadu_epi32(__W, __U, __P); 
}

__m128i test_mm_maskz_loadu_epi32(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_loadu_epi32
  // CHECK: @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maskz_loadu_epi32(__U, __P); 
}

__m256i test_mm256_loadu_epi32(void const *__P) {
  // CHECK-LABEL: @test_mm256_loadu_epi32
  // CHECK: load <4 x i64>, <4 x i64>* %{{.*}}, align 1{{$}}
  return _mm256_loadu_epi32(__P);
}

__m256i test_mm256_mask_loadu_epi32(__m256i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_loadu_epi32
  // CHECK: @llvm.masked.load.v8i32.p0v8i32(<8 x i32>* %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_mask_loadu_epi32(__W, __U, __P); 
}

__m256i test_mm256_maskz_loadu_epi32(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_loadu_epi32
  // CHECK: @llvm.masked.load.v8i32.p0v8i32(<8 x i32>* %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_maskz_loadu_epi32(__U, __P); 
}

__m128d test_mm_mask_loadu_pd(__m128d __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_loadu_pd
  // CHECK: @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %{{.*}}, i32 1, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_mask_loadu_pd(__W, __U, __P); 
}

__m128d test_mm_maskz_loadu_pd(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_loadu_pd
  // CHECK: @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %{{.*}}, i32 1, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_maskz_loadu_pd(__U, __P); 
}

__m256d test_mm256_mask_loadu_pd(__m256d __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_loadu_pd
  // CHECK: @llvm.masked.load.v4f64.p0v4f64(<4 x double>* %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_mask_loadu_pd(__W, __U, __P); 
}

__m256d test_mm256_maskz_loadu_pd(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_loadu_pd
  // CHECK: @llvm.masked.load.v4f64.p0v4f64(<4 x double>* %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_maskz_loadu_pd(__U, __P); 
}

__m128 test_mm_mask_loadu_ps(__m128 __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_loadu_ps
  // CHECK: @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_mask_loadu_ps(__W, __U, __P); 
}

__m128 test_mm_maskz_loadu_ps(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_loadu_ps
  // CHECK: @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_maskz_loadu_ps(__U, __P); 
}

__m256 test_mm256_mask_loadu_ps(__m256 __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_loadu_ps
  // CHECK: @llvm.masked.load.v8f32.p0v8f32(<8 x float>* %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_mask_loadu_ps(__W, __U, __P); 
}

__m256 test_mm256_maskz_loadu_ps(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_loadu_ps
  // CHECK: @llvm.masked.load.v8f32.p0v8f32(<8 x float>* %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_maskz_loadu_ps(__U, __P); 
}

void test_mm_mask_store_pd(void *__P, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_store_pd
  // CHECK: @llvm.masked.store.v2f64.p0v2f64(<2 x double> %{{.*}}, <2 x double>* %{{.*}}, i32 16, <2 x i1> %{{.*}})
  return _mm_mask_store_pd(__P, __U, __A); 
}

void test_mm256_mask_store_pd(void *__P, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_store_pd
  // CHECK: @llvm.masked.store.v4f64.p0v4f64(<4 x double> %{{.*}}, <4 x double>* %{{.*}}, i32 32, <4 x i1> %{{.*}})
  return _mm256_mask_store_pd(__P, __U, __A); 
}

void test_mm_mask_store_ps(void *__P, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_store_ps
  // CHECK: @llvm.masked.store.v4f32.p0v4f32(<4 x float> %{{.*}}, <4 x float>* %{{.*}}, i32 16, <4 x i1> %{{.*}})
  return _mm_mask_store_ps(__P, __U, __A); 
}

void test_mm256_mask_store_ps(void *__P, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_store_ps
  // CHECK: @llvm.masked.store.v8f32.p0v8f32(<8 x float> %{{.*}}, <8 x float>* %{{.*}}, i32 32, <8 x i1> %{{.*}})
  return _mm256_mask_store_ps(__P, __U, __A); 
}

void test_mm_storeu_epi64(void *__p, __m128i __a) {
  // check-label: @test_mm_storeu_epi64
  // check: store <2 x i64> %{{.*}}, <2 x i64>* %{{.*}}, align 1{{$}}
  return _mm_storeu_epi64(__p, __a);
}

void test_mm_mask_storeu_epi64(void *__P, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_storeu_epi64
  // CHECK: @llvm.masked.store.v2i64.p0v2i64(<2 x i64> %{{.*}}, <2 x i64>* %{{.*}}, i32 1, <2 x i1> %{{.*}})
  return _mm_mask_storeu_epi64(__P, __U, __A); 
}

void test_mm256_storeu_epi64(void *__P, __m256i __A) {
  // CHECK-LABEL: @test_mm256_storeu_epi64
  // CHECK: store <4 x i64> %{{.*}}, <4 x i64>* %{{.*}}, align 1{{$}}
  return _mm256_storeu_epi64(__P, __A);
}

void test_mm256_mask_storeu_epi64(void *__P, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_storeu_epi64
  // CHECK: @llvm.masked.store.v4i64.p0v4i64(<4 x i64> %{{.*}}, <4 x i64>* %{{.*}}, i32 1, <4 x i1> %{{.*}})
  return _mm256_mask_storeu_epi64(__P, __U, __A); 
}

void test_mm_storeu_epi32(void *__P, __m128i __A) {
  // CHECK-LABEL: @test_mm_storeu_epi32
  // CHECK: store <2 x i64> %{{.*}}, <2 x i64>* %{{.*}}, align 1{{$}}
  return _mm_storeu_epi32(__P, __A);
}

void test_mm_mask_storeu_epi32(void *__P, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_storeu_epi32
  // CHECK: @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %{{.*}}, <4 x i32>* %{{.*}}, i32 1, <4 x i1> %{{.*}})
  return _mm_mask_storeu_epi32(__P, __U, __A); 
}

void test_mm256_storeu_epi32(void *__P, __m256i __A) {
  // CHECK-LABEL: @test_mm256_storeu_epi32
  // CHECK: store <4 x i64> %{{.*}}, <4 x i64>* %{{.*}}, align 1{{$}}
  return _mm256_storeu_epi32(__P, __A);
}

void test_mm256_mask_storeu_epi32(void *__P, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_storeu_epi32
  // CHECK: @llvm.masked.store.v8i32.p0v8i32(<8 x i32> %{{.*}}, <8 x i32>* %{{.*}}, i32 1, <8 x i1> %{{.*}})
  return _mm256_mask_storeu_epi32(__P, __U, __A); 
}

void test_mm_mask_storeu_pd(void *__P, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_storeu_pd
  // CHECK: @llvm.masked.store.v2f64.p0v2f64(<2 x double> %{{.*}}, <2 x double>* %{{.*}}, i32 1, <2 x i1> %{{.*}})
  return _mm_mask_storeu_pd(__P, __U, __A); 
}

void test_mm256_mask_storeu_pd(void *__P, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_storeu_pd
  // CHECK: @llvm.masked.store.v4f64.p0v4f64(<4 x double> %{{.*}}, <4 x double>* %{{.*}}, i32 1, <4 x i1> %{{.*}})
  return _mm256_mask_storeu_pd(__P, __U, __A); 
}

void test_mm_mask_storeu_ps(void *__P, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_storeu_ps
  // CHECK: @llvm.masked.store.v4f32.p0v4f32(<4 x float> %{{.*}}, <4 x float>* %{{.*}}, i32 1, <4 x i1> %{{.*}})
  return _mm_mask_storeu_ps(__P, __U, __A); 
}

void test_mm256_mask_storeu_ps(void *__P, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_storeu_ps
  // CHECK: @llvm.masked.store.v8f32.p0v8f32(<8 x float> %{{.*}}, <8 x float>* %{{.*}}, i32 1, <8 x i1> %{{.*}})
  return _mm256_mask_storeu_ps(__P, __U, __A); 
}

__m128d test_mm_mask_unpackhi_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_unpackhi_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 3>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_unpackhi_pd(__W, __U, __A, __B); 
}

__m128d test_mm_maskz_unpackhi_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_unpackhi_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 3>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_unpackhi_pd(__U, __A, __B); 
}

__m256d test_mm256_mask_unpackhi_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_unpackhi_pd
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}} <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_unpackhi_pd(__W, __U, __A, __B); 
}

__m256d test_mm256_maskz_unpackhi_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpackhi_pd
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}} <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_unpackhi_pd(__U, __A, __B); 
}

__m128 test_mm_mask_unpackhi_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_unpackhi_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}} <4 x float> %{{.*}}
  return _mm_mask_unpackhi_ps(__W, __U, __A, __B); 
}

__m128 test_mm_maskz_unpackhi_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_unpackhi_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}} <4 x float> %{{.*}}
  return _mm_maskz_unpackhi_ps(__U, __A, __B); 
}

__m256 test_mm256_mask_unpackhi_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_unpackhi_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_unpackhi_ps(__W, __U, __A, __B); 
}

__m256 test_mm256_maskz_unpackhi_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpackhi_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_unpackhi_ps(__U, __A, __B); 
}

__m128d test_mm_mask_unpacklo_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_unpacklo_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 2>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_unpacklo_pd(__W, __U, __A, __B); 
}

__m128d test_mm_maskz_unpacklo_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_unpacklo_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 2>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_unpacklo_pd(__U, __A, __B); 
}

__m256d test_mm256_mask_unpacklo_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_unpacklo_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  // CHECK: select <4 x i1> %{{.*}} <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_unpacklo_pd(__W, __U, __A, __B); 
}

__m256d test_mm256_maskz_unpacklo_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpacklo_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  // CHECK: select <4 x i1> %{{.*}} <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_unpacklo_pd(__U, __A, __B); 
}

__m128 test_mm_mask_unpacklo_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_unpacklo_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  // CHECK: select <4 x i1> %{{.*}} <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_unpacklo_ps(__W, __U, __A, __B); 
}

__m128 test_mm_maskz_unpacklo_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_unpacklo_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  // CHECK: select <4 x i1> %{{.*}} <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_unpacklo_ps(__U, __A, __B); 
}

__m256 test_mm256_mask_unpacklo_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_unpacklo_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_unpacklo_ps(__W, __U, __A, __B); 
}

__m256 test_mm256_maskz_unpacklo_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpacklo_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_unpacklo_ps(__U, __A, __B); 
}

__m128d test_mm_rcp14_pd(__m128d __A) {
  // CHECK-LABEL: @test_mm_rcp14_pd
  // CHECK: @llvm.x86.avx512.rcp14.pd.128
  return _mm_rcp14_pd(__A); 
}

__m128d test_mm_mask_rcp14_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_rcp14_pd
  // CHECK: @llvm.x86.avx512.rcp14.pd.128
  return _mm_mask_rcp14_pd(__W, __U, __A); 
}

__m128d test_mm_maskz_rcp14_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_rcp14_pd
  // CHECK: @llvm.x86.avx512.rcp14.pd.128
  return _mm_maskz_rcp14_pd(__U, __A); 
}

__m256d test_mm256_rcp14_pd(__m256d __A) {
  // CHECK-LABEL: @test_mm256_rcp14_pd
  // CHECK: @llvm.x86.avx512.rcp14.pd.256
  return _mm256_rcp14_pd(__A); 
}

__m256d test_mm256_mask_rcp14_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_rcp14_pd
  // CHECK: @llvm.x86.avx512.rcp14.pd.256
  return _mm256_mask_rcp14_pd(__W, __U, __A); 
}

__m256d test_mm256_maskz_rcp14_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_rcp14_pd
  // CHECK: @llvm.x86.avx512.rcp14.pd.256
  return _mm256_maskz_rcp14_pd(__U, __A); 
}

__m128 test_mm_rcp14_ps(__m128 __A) {
  // CHECK-LABEL: @test_mm_rcp14_ps
  // CHECK: @llvm.x86.avx512.rcp14.ps.128
  return _mm_rcp14_ps(__A); 
}

__m128 test_mm_mask_rcp14_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_rcp14_ps
  // CHECK: @llvm.x86.avx512.rcp14.ps.128
  return _mm_mask_rcp14_ps(__W, __U, __A); 
}

__m128 test_mm_maskz_rcp14_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_rcp14_ps
  // CHECK: @llvm.x86.avx512.rcp14.ps.128
  return _mm_maskz_rcp14_ps(__U, __A); 
}

__m256 test_mm256_rcp14_ps(__m256 __A) {
  // CHECK-LABEL: @test_mm256_rcp14_ps
  // CHECK: @llvm.x86.avx512.rcp14.ps.256
  return _mm256_rcp14_ps(__A); 
}

__m256 test_mm256_mask_rcp14_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_rcp14_ps
  // CHECK: @llvm.x86.avx512.rcp14.ps.256
  return _mm256_mask_rcp14_ps(__W, __U, __A); 
}

__m256 test_mm256_maskz_rcp14_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_rcp14_ps
  // CHECK: @llvm.x86.avx512.rcp14.ps.256
  return _mm256_maskz_rcp14_ps(__U, __A); 
}

__m128d test_mm_mask_permute_pd(__m128d __W, __mmask8 __U, __m128d __X) {
  // CHECK-LABEL: @test_mm_mask_permute_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_permute_pd(__W, __U, __X, 1); 
}

__m128d test_mm_maskz_permute_pd(__mmask8 __U, __m128d __X) {
  // CHECK-LABEL: @test_mm_maskz_permute_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_permute_pd(__U, __X, 1); 
}

__m256d test_mm256_mask_permute_pd(__m256d __W, __mmask8 __U, __m256d __X) {
  // CHECK-LABEL: @test_mm256_mask_permute_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_permute_pd(__W, __U, __X, 5); 
}

__m256d test_mm256_maskz_permute_pd(__mmask8 __U, __m256d __X) {
  // CHECK-LABEL: @test_mm256_maskz_permute_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_permute_pd(__U, __X, 5); 
}

__m128 test_mm_mask_permute_ps(__m128 __W, __mmask8 __U, __m128 __X) {
  // CHECK-LABEL: @test_mm_mask_permute_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_permute_ps(__W, __U, __X, 0x1b); 
}

__m128 test_mm_maskz_permute_ps(__mmask8 __U, __m128 __X) {
  // CHECK-LABEL: @test_mm_maskz_permute_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_permute_ps(__U, __X, 0x1b); 
}

__m256 test_mm256_mask_permute_ps(__m256 __W, __mmask8 __U, __m256 __X) {
  // CHECK-LABEL: @test_mm256_mask_permute_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> poison, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_permute_ps(__W, __U, __X, 0x1b); 
}

__m256 test_mm256_maskz_permute_ps(__mmask8 __U, __m256 __X) {
  // CHECK-LABEL: @test_mm256_maskz_permute_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> poison, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_permute_ps(__U, __X, 0x1b); 
}

__m128d test_mm_mask_permutevar_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_permutevar_pd
  // CHECK: @llvm.x86.avx.vpermilvar.pd
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_permutevar_pd(__W, __U, __A, __C); 
}

__m128d test_mm_maskz_permutevar_pd(__mmask8 __U, __m128d __A, __m128i __C) {
  // CHECK-LABEL: @test_mm_maskz_permutevar_pd
  // CHECK: @llvm.x86.avx.vpermilvar.pd
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_permutevar_pd(__U, __A, __C); 
}

__m256d test_mm256_mask_permutevar_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256i __C) {
  // CHECK-LABEL: @test_mm256_mask_permutevar_pd
  // CHECK: @llvm.x86.avx.vpermilvar.pd.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_permutevar_pd(__W, __U, __A, __C); 
}

__m256d test_mm256_maskz_permutevar_pd(__mmask8 __U, __m256d __A, __m256i __C) {
  // CHECK-LABEL: @test_mm256_maskz_permutevar_pd
  // CHECK: @llvm.x86.avx.vpermilvar.pd.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_permutevar_pd(__U, __A, __C); 
}

__m128 test_mm_mask_permutevar_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_permutevar_ps
  // CHECK: @llvm.x86.avx.vpermilvar.ps
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_permutevar_ps(__W, __U, __A, __C); 
}

__m128 test_mm_maskz_permutevar_ps(__mmask8 __U, __m128 __A, __m128i __C) {
  // CHECK-LABEL: @test_mm_maskz_permutevar_ps
  // CHECK: @llvm.x86.avx.vpermilvar.ps
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_permutevar_ps(__U, __A, __C); 
}

__m256 test_mm256_mask_permutevar_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256i __C) {
  // CHECK-LABEL: @test_mm256_mask_permutevar_ps
  // CHECK: @llvm.x86.avx.vpermilvar.ps.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_permutevar_ps(__W, __U, __A, __C); 
}

__m256 test_mm256_maskz_permutevar_ps(__mmask8 __U, __m256 __A, __m256i __C) {
  // CHECK-LABEL: @test_mm256_maskz_permutevar_ps
  // CHECK: @llvm.x86.avx.vpermilvar.ps.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_permutevar_ps(__U, __A, __C); 
}

__mmask8 test_mm_test_epi32_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_test_epi32_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <4 x i32> %{{.*}}, %{{.*}}
  return _mm_test_epi32_mask(__A, __B); 
}

__mmask8 test_mm_mask_test_epi32_mask(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_test_epi32_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_test_epi32_mask(__U, __A, __B); 
}

__mmask8 test_mm256_test_epi32_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_test_epi32_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <8 x i32> %{{.*}}, %{{.*}}
  return _mm256_test_epi32_mask(__A, __B); 
}

__mmask8 test_mm256_mask_test_epi32_mask(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_test_epi32_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_test_epi32_mask(__U, __A, __B); 
}

__mmask8 test_mm_test_epi64_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_test_epi64_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <2 x i64> %{{.*}}, %{{.*}}
  return _mm_test_epi64_mask(__A, __B); 
}

__mmask8 test_mm_mask_test_epi64_mask(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_test_epi64_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_test_epi64_mask(__U, __A, __B); 
}

__mmask8 test_mm256_test_epi64_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_test_epi64_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <4 x i64> %{{.*}}, %{{.*}}
  return _mm256_test_epi64_mask(__A, __B); 
}

__mmask8 test_mm256_mask_test_epi64_mask(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_test_epi64_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_test_epi64_mask(__U, __A, __B); 
}

__mmask8 test_mm_testn_epi32_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_testn_epi32_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  return _mm_testn_epi32_mask(__A, __B); 
}

__mmask8 test_mm_mask_testn_epi32_mask(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_testn_epi32_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_testn_epi32_mask(__U, __A, __B); 
}

__mmask8 test_mm256_testn_epi32_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_testn_epi32_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  return _mm256_testn_epi32_mask(__A, __B); 
}

__mmask8 test_mm256_mask_testn_epi32_mask(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_testn_epi32_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_testn_epi32_mask(__U, __A, __B); 
}

__mmask8 test_mm_testn_epi64_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_testn_epi64_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  return _mm_testn_epi64_mask(__A, __B); 
}

__mmask8 test_mm_mask_testn_epi64_mask(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_testn_epi64_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_testn_epi64_mask(__U, __A, __B); 
}

__mmask8 test_mm256_testn_epi64_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_testn_epi64_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  return _mm256_testn_epi64_mask(__A, __B); 
}

__mmask8 test_mm256_mask_testn_epi64_mask(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_testn_epi64_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_testn_epi64_mask(__U, __A, __B); 
}

__m128i test_mm_mask_unpackhi_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_unpackhi_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_unpackhi_epi32(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpackhi_epi32(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_unpackhi_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_unpackhi_epi32(__U, __A, __B); 
}

__m256i test_mm256_mask_unpackhi_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_unpackhi_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_unpackhi_epi32(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpackhi_epi32(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpackhi_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_unpackhi_epi32(__U, __A, __B); 
}

__m128i test_mm_mask_unpackhi_epi64(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_unpackhi_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 3>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_unpackhi_epi64(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpackhi_epi64(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_unpackhi_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 3>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_unpackhi_epi64(__U, __A, __B); 
}

__m256i test_mm256_mask_unpackhi_epi64(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_unpackhi_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_unpackhi_epi64(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpackhi_epi64(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpackhi_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_unpackhi_epi64(__U, __A, __B); 
}

__m128i test_mm_mask_unpacklo_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_unpacklo_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_unpacklo_epi32(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpacklo_epi32(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_unpacklo_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_unpacklo_epi32(__U, __A, __B); 
}

__m256i test_mm256_mask_unpacklo_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_unpacklo_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_unpacklo_epi32(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpacklo_epi32(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpacklo_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_unpacklo_epi32(__U, __A, __B); 
}

__m128i test_mm_mask_unpacklo_epi64(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_unpacklo_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 0, i32 2>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_unpacklo_epi64(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpacklo_epi64(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_unpacklo_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 0, i32 2>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_unpacklo_epi64(__U, __A, __B); 
}

__m256i test_mm256_mask_unpacklo_epi64(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_unpacklo_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_unpacklo_epi64(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpacklo_epi64(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpacklo_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_unpacklo_epi64(__U, __A, __B); 
}

__m128i test_mm_mask_sra_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_sra_epi32
  // CHECK: @llvm.x86.sse2.psra.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_sra_epi32(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_sra_epi32(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_sra_epi32
  // CHECK: @llvm.x86.sse2.psra.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_sra_epi32(__U, __A, __B); 
}

__m256i test_mm256_mask_sra_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_mask_sra_epi32
  // CHECK: @llvm.x86.avx2.psra.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_sra_epi32(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_sra_epi32(__mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_sra_epi32
  // CHECK: @llvm.x86.avx2.psra.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_sra_epi32(__U, __A, __B); 
}

__m128i test_mm_mask_srai_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_srai_epi32
  // CHECK: @llvm.x86.sse2.psrai.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_srai_epi32(__W, __U, __A, 5); 
}

__m128i test_mm_mask_srai_epi32_2(__m128i __W, __mmask8 __U, __m128i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm_mask_srai_epi32_2
  // CHECK: @llvm.x86.sse2.psrai.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_srai_epi32(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_srai_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_srai_epi32
  // CHECK: @llvm.x86.sse2.psrai.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_srai_epi32(__U, __A, 5); 
}

__m128i test_mm_maskz_srai_epi32_2(__mmask8 __U, __m128i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm_maskz_srai_epi32_2
  // CHECK: @llvm.x86.sse2.psrai.d
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_srai_epi32(__U, __A, __B); 
}

__m256i test_mm256_mask_srai_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_srai_epi32
  // CHECK: @llvm.x86.avx2.psrai.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_srai_epi32(__W, __U, __A, 5); 
}

__m256i test_mm256_mask_srai_epi32_2(__m256i __W, __mmask8 __U, __m256i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm256_mask_srai_epi32_2
  // CHECK: @llvm.x86.avx2.psrai.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_srai_epi32(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_srai_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_srai_epi32
  // CHECK: @llvm.x86.avx2.psrai.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_srai_epi32(__U, __A, 5); 
}

__m256i test_mm256_maskz_srai_epi32_2(__mmask8 __U, __m256i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm256_maskz_srai_epi32_2
  // CHECK: @llvm.x86.avx2.psrai.d
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_srai_epi32(__U, __A, __B); 
}

__m128i test_mm_sra_epi64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_sra_epi64
  // CHECK: @llvm.x86.avx512.psra.q.128
  return _mm_sra_epi64(__A, __B); 
}

__m128i test_mm_mask_sra_epi64(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_sra_epi64
  // CHECK: @llvm.x86.avx512.psra.q.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_sra_epi64(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_sra_epi64(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_sra_epi64
  // CHECK: @llvm.x86.avx512.psra.q.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_sra_epi64(__U, __A, __B); 
}

__m256i test_mm256_sra_epi64(__m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_sra_epi64
  // CHECK: @llvm.x86.avx512.psra.q.256
  return _mm256_sra_epi64(__A, __B); 
}

__m256i test_mm256_mask_sra_epi64(__m256i __W, __mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_mask_sra_epi64
  // CHECK: @llvm.x86.avx512.psra.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_sra_epi64(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_sra_epi64(__mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_sra_epi64
  // CHECK: @llvm.x86.avx512.psra.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_sra_epi64(__U, __A, __B); 
}

__m128i test_mm_srai_epi64(__m128i __A) {
  // CHECK-LABEL: @test_mm_srai_epi64
  // CHECK: @llvm.x86.avx512.psrai.q.128
  return _mm_srai_epi64(__A, 5); 
}

__m128i test_mm_srai_epi64_2(__m128i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm_srai_epi64_2
  // CHECK: @llvm.x86.avx512.psrai.q.128
  return _mm_srai_epi64(__A, __B); 
}

__m128i test_mm_mask_srai_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_srai_epi64
  // CHECK: @llvm.x86.avx512.psrai.q.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_srai_epi64(__W, __U, __A, 5); 
}

__m128i test_mm_mask_srai_epi64_2(__m128i __W, __mmask8 __U, __m128i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm_mask_srai_epi64_2
  // CHECK: @llvm.x86.avx512.psrai.q.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_srai_epi64(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_srai_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_srai_epi64
  // CHECK: @llvm.x86.avx512.psrai.q.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_srai_epi64(__U, __A, 5); 
}

__m128i test_mm_maskz_srai_epi64_2(__mmask8 __U, __m128i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm_maskz_srai_epi64_2
  // CHECK: @llvm.x86.avx512.psrai.q.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_srai_epi64(__U, __A, __B); 
}

__m256i test_mm256_srai_epi64(__m256i __A) {
  // CHECK-LABEL: @test_mm256_srai_epi64
  // CHECK: @llvm.x86.avx512.psrai.q.256
  return _mm256_srai_epi64(__A, 5); 
}

__m256i test_mm256_srai_epi64_2(__m256i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm256_srai_epi64_2
  // CHECK: @llvm.x86.avx512.psrai.q.256
  return _mm256_srai_epi64(__A, __B); 
}

__m256i test_mm256_mask_srai_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_srai_epi64
  // CHECK: @llvm.x86.avx512.psrai.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_srai_epi64(__W, __U, __A, 5); 
}

__m256i test_mm256_mask_srai_epi64_2(__m256i __W, __mmask8 __U, __m256i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm256_mask_srai_epi64_2
  // CHECK: @llvm.x86.avx512.psrai.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_srai_epi64(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_srai_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_srai_epi64
  // CHECK: @llvm.x86.avx512.psrai.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_srai_epi64(__U, __A, 5); 
}

__m256i test_mm256_maskz_srai_epi64_2(__mmask8 __U, __m256i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm256_maskz_srai_epi64_2
  // CHECK: @llvm.x86.avx512.psrai.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_srai_epi64(__U, __A, __B); 
}

__m128i test_mm_ternarylogic_epi32(__m128i __A, __m128i __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.pternlog.d.128
  return _mm_ternarylogic_epi32(__A, __B, __C, 4); 
}

__m128i test_mm_mask_ternarylogic_epi32(__m128i __A, __mmask8 __U, __m128i __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.pternlog.d.128
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_ternarylogic_epi32(__A, __U, __B, __C, 4); 
}

__m128i test_mm_maskz_ternarylogic_epi32(__mmask8 __U, __m128i __A, __m128i __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_maskz_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.pternlog.d.128
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> zeroinitializer
  return _mm_maskz_ternarylogic_epi32(__U, __A, __B, __C, 4); 
}

__m256i test_mm256_ternarylogic_epi32(__m256i __A, __m256i __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.pternlog.d.256
  return _mm256_ternarylogic_epi32(__A, __B, __C, 4); 
}

__m256i test_mm256_mask_ternarylogic_epi32(__m256i __A, __mmask8 __U, __m256i __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_mask_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.pternlog.d.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_ternarylogic_epi32(__A, __U, __B, __C, 4); 
}

__m256i test_mm256_maskz_ternarylogic_epi32(__mmask8 __U, __m256i __A, __m256i __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_maskz_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.pternlog.d.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> zeroinitializer
  return _mm256_maskz_ternarylogic_epi32(__U, __A, __B, __C, 4); 
}

__m128i test_mm_ternarylogic_epi64(__m128i __A, __m128i __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.pternlog.q.128
  return _mm_ternarylogic_epi64(__A, __B, __C, 4); 
}

__m128i test_mm_mask_ternarylogic_epi64(__m128i __A, __mmask8 __U, __m128i __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.pternlog.q.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_ternarylogic_epi64(__A, __U, __B, __C, 4); 
}

__m128i test_mm_maskz_ternarylogic_epi64(__mmask8 __U, __m128i __A, __m128i __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_maskz_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.pternlog.q.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> zeroinitializer
  return _mm_maskz_ternarylogic_epi64(__U, __A, __B, __C, 4); 
}

__m256i test_mm256_ternarylogic_epi64(__m256i __A, __m256i __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.pternlog.q.256
  return _mm256_ternarylogic_epi64(__A, __B, __C, 4); 
}

__m256i test_mm256_mask_ternarylogic_epi64(__m256i __A, __mmask8 __U, __m256i __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_mask_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.pternlog.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_ternarylogic_epi64(__A, __U, __B, __C, 4); 
}

__m256i test_mm256_maskz_ternarylogic_epi64(__mmask8 __U, __m256i __A, __m256i __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_maskz_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.pternlog.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> zeroinitializer
  return _mm256_maskz_ternarylogic_epi64(__U, __A, __B, __C, 4); 
}
__m256 test_mm256_shuffle_f32x4(__m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_shuffle_f32x4
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  return _mm256_shuffle_f32x4(__A, __B, 3); 
}

__m256 test_mm256_mask_shuffle_f32x4(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_shuffle_f32x4
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_shuffle_f32x4(__W, __U, __A, __B, 3); 
}

__m256 test_mm256_maskz_shuffle_f32x4(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_shuffle_f32x4
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_shuffle_f32x4(__U, __A, __B, 3); 
}

__m256d test_mm256_shuffle_f64x2(__m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_shuffle_f64x2
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  return _mm256_shuffle_f64x2(__A, __B, 3); 
}

__m256d test_mm256_mask_shuffle_f64x2(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_shuffle_f64x2
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_shuffle_f64x2(__W, __U, __A, __B, 3); 
}

__m256d test_mm256_maskz_shuffle_f64x2(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_shuffle_f64x2
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_shuffle_f64x2(__U, __A, __B, 3); 
}

__m256i test_mm256_shuffle_i32x4(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_shuffle_i32x4
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  return _mm256_shuffle_i32x4(__A, __B, 3); 
}

__m256i test_mm256_mask_shuffle_i32x4(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_shuffle_i32x4
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_shuffle_i32x4(__W, __U, __A, __B, 3); 
}

__m256i test_mm256_maskz_shuffle_i32x4(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_shuffle_i32x4
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_shuffle_i32x4(__U, __A, __B, 3); 
}

__m256i test_mm256_shuffle_i64x2(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_shuffle_i64x2
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  return _mm256_shuffle_i64x2(__A, __B, 3); 
}

__m256i test_mm256_mask_shuffle_i64x2(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_shuffle_i64x2
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_shuffle_i64x2(__W, __U, __A, __B, 3); 
}

__m256i test_mm256_maskz_shuffle_i64x2(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_shuffle_i64x2
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_shuffle_i64x2(__U, __A, __B, 3); 
}

__m128d test_mm_mask_shuffle_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_shuffle_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 3>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_shuffle_pd(__W, __U, __A, __B, 3); 
}

__m128d test_mm_maskz_shuffle_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_shuffle_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 3>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_shuffle_pd(__U, __A, __B, 3); 
}

__m256d test_mm256_mask_shuffle_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_shuffle_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 1, i32 5, i32 2, i32 6>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_shuffle_pd(__W, __U, __A, __B, 3); 
}

__m256d test_mm256_maskz_shuffle_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_shuffle_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 1, i32 5, i32 2, i32 6>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_shuffle_pd(__U, __A, __B, 3); 
}

__m128 test_mm_mask_shuffle_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_shuffle_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 4>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_shuffle_ps(__W, __U, __A, __B, 4); 
}

__m128 test_mm_maskz_shuffle_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_shuffle_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 4>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_shuffle_ps(__U, __A, __B, 4); 
}

__m256 test_mm256_mask_shuffle_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_shuffle_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 8, i32 8, i32 4, i32 5, i32 12, i32 12>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_shuffle_ps(__W, __U, __A, __B, 4); 
}

__m256 test_mm256_maskz_shuffle_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_shuffle_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 8, i32 8, i32 4, i32 5, i32 12, i32 12>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_shuffle_ps(__U, __A, __B, 4); 
}

__m128d test_mm_rsqrt14_pd(__m128d __A) {
  // CHECK-LABEL: @test_mm_rsqrt14_pd
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.128
  return _mm_rsqrt14_pd(__A); 
}

__m128d test_mm_mask_rsqrt14_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_rsqrt14_pd
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.128
  return _mm_mask_rsqrt14_pd(__W, __U, __A); 
}

__m128d test_mm_maskz_rsqrt14_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_rsqrt14_pd
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.128
  return _mm_maskz_rsqrt14_pd(__U, __A); 
}

__m256d test_mm256_rsqrt14_pd(__m256d __A) {
  // CHECK-LABEL: @test_mm256_rsqrt14_pd
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.256
  return _mm256_rsqrt14_pd(__A); 
}

__m256d test_mm256_mask_rsqrt14_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_rsqrt14_pd
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.256
  return _mm256_mask_rsqrt14_pd(__W, __U, __A); 
}

__m256d test_mm256_maskz_rsqrt14_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_rsqrt14_pd
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.256
  return _mm256_maskz_rsqrt14_pd(__U, __A); 
}

__m128 test_mm_rsqrt14_ps(__m128 __A) {
  // CHECK-LABEL: @test_mm_rsqrt14_ps
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.128
  return _mm_rsqrt14_ps(__A); 
}

__m128 test_mm_mask_rsqrt14_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_rsqrt14_ps
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.128
  return _mm_mask_rsqrt14_ps(__W, __U, __A); 
}

__m128 test_mm_maskz_rsqrt14_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_rsqrt14_ps
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.128
  return _mm_maskz_rsqrt14_ps(__U, __A); 
}

__m256 test_mm256_rsqrt14_ps(__m256 __A) {
  // CHECK-LABEL: @test_mm256_rsqrt14_ps
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.256
  return _mm256_rsqrt14_ps(__A); 
}

__m256 test_mm256_mask_rsqrt14_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_rsqrt14_ps
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.256
  return _mm256_mask_rsqrt14_ps(__W, __U, __A); 
}

__m256 test_mm256_maskz_rsqrt14_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_rsqrt14_ps
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.256
  return _mm256_maskz_rsqrt14_ps(__U, __A); 
}

__m256 test_mm256_broadcast_f32x4(__m128 __A) {
  // CHECK-LABEL: @test_mm256_broadcast_f32x4
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  return _mm256_broadcast_f32x4(__A); 
}

__m256 test_mm256_mask_broadcast_f32x4(__m256 __O, __mmask8 __M, __m128 __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcast_f32x4
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_broadcast_f32x4(__O, __M, __A); 
}

__m256 test_mm256_maskz_broadcast_f32x4(__mmask8 __M, __m128 __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcast_f32x4
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_broadcast_f32x4(__M, __A); 
}

__m256i test_mm256_broadcast_i32x4(__m128i const* __A) {
  // CHECK-LABEL: @test_mm256_broadcast_i32x4
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  return _mm256_broadcast_i32x4(_mm_loadu_si128(__A)); 
}

__m256i test_mm256_mask_broadcast_i32x4(__m256i __O, __mmask8 __M, __m128i const* __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcast_i32x4
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_broadcast_i32x4(__O, __M, _mm_loadu_si128(__A)); 
}

__m256i test_mm256_maskz_broadcast_i32x4(__mmask8 __M, __m128i const* __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcast_i32x4
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_broadcast_i32x4(__M, _mm_loadu_si128(__A)); 
}

__m256d test_mm256_mask_broadcastsd_pd(__m256d __O, __mmask8 __M, __m128d __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcastsd_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_broadcastsd_pd(__O, __M, __A);
}

__m256d test_mm256_maskz_broadcastsd_pd(__mmask8 __M, __m128d __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcastsd_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_broadcastsd_pd(__M, __A);
}

__m128 test_mm_mask_broadcastss_ps(__m128 __O, __mmask8 __M, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_broadcastss_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_broadcastss_ps(__O, __M, __A);
}

__m128 test_mm_maskz_broadcastss_ps(__mmask8 __M, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_broadcastss_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_broadcastss_ps(__M, __A);
}

__m256 test_mm256_mask_broadcastss_ps(__m256 __O, __mmask8 __M, __m128 __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcastss_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_broadcastss_ps(__O, __M, __A);
}

__m256 test_mm256_maskz_broadcastss_ps(__mmask8 __M, __m128 __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcastss_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_broadcastss_ps(__M, __A);
}

__m128i test_mm_mask_broadcastd_epi32(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_broadcastd_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_broadcastd_epi32(__O, __M, __A);
}

__m128i test_mm_maskz_broadcastd_epi32(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_broadcastd_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_broadcastd_epi32(__M, __A);
}

__m256i test_mm256_mask_broadcastd_epi32(__m256i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcastd_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_broadcastd_epi32(__O, __M, __A);
}

__m256i test_mm256_maskz_broadcastd_epi32(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcastd_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_broadcastd_epi32(__M, __A);
}

__m128i test_mm_mask_broadcastq_epi64(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_broadcastq_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> zeroinitializer
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_broadcastq_epi64(__O, __M, __A);
}

__m128i test_mm_maskz_broadcastq_epi64(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_broadcastq_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> zeroinitializer
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_broadcastq_epi64(__M, __A);
}

__m256i test_mm256_mask_broadcastq_epi64(__m256i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcastq_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_broadcastq_epi64(__O, __M, __A);
}

__m256i test_mm256_maskz_broadcastq_epi64(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcastq_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_broadcastq_epi64(__M, __A);
}

__m128i test_mm_cvtsepi32_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.128
  return _mm_cvtsepi32_epi8(__A); 
}

__m128i test_mm_mask_cvtsepi32_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.128
  return _mm_mask_cvtsepi32_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtsepi32_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.128
  return _mm_maskz_cvtsepi32_epi8(__M, __A); 
}

void test_mm_mask_cvtsepi32_storeu_epi8(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.mem.128
  return _mm_mask_cvtsepi32_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm256_cvtsepi32_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.256
  return _mm256_cvtsepi32_epi8(__A); 
}

__m128i test_mm256_mask_cvtsepi32_epi8(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.256
  return _mm256_mask_cvtsepi32_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtsepi32_epi8(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.256
  return _mm256_maskz_cvtsepi32_epi8(__M, __A); 
}

void test_mm256_mask_cvtsepi32_storeu_epi8(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.mem.256
  return _mm256_mask_cvtsepi32_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm_cvtsepi32_epi16(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.128
  return _mm_cvtsepi32_epi16(__A); 
}

__m128i test_mm_mask_cvtsepi32_epi16(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.128
  return _mm_mask_cvtsepi32_epi16(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtsepi32_epi16(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.128
  return _mm_maskz_cvtsepi32_epi16(__M, __A); 
}

void test_mm_mask_cvtsepi32_storeu_epi16(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.mem.128
  return _mm_mask_cvtsepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm256_cvtsepi32_epi16(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.256
  return _mm256_cvtsepi32_epi16(__A); 
}

__m128i test_mm256_mask_cvtsepi32_epi16(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.256
  return _mm256_mask_cvtsepi32_epi16(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtsepi32_epi16(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.256
  return _mm256_maskz_cvtsepi32_epi16(__M, __A); 
}

void test_mm256_mask_cvtsepi32_storeu_epi16(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.mem.256
  return _mm256_mask_cvtsepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm_cvtsepi64_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.128
  return _mm_cvtsepi64_epi8(__A); 
}

__m128i test_mm_mask_cvtsepi64_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.128
  return _mm_mask_cvtsepi64_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtsepi64_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.128
  return _mm_maskz_cvtsepi64_epi8(__M, __A); 
}

void test_mm_mask_cvtsepi64_storeu_epi8(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.mem.128
  return _mm_mask_cvtsepi64_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm256_cvtsepi64_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.256
  return _mm256_cvtsepi64_epi8(__A); 
}

__m128i test_mm256_mask_cvtsepi64_epi8(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.256
  return _mm256_mask_cvtsepi64_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtsepi64_epi8(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.256
  return _mm256_maskz_cvtsepi64_epi8(__M, __A); 
}

void test_mm256_mask_cvtsepi64_storeu_epi8(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.mem.256
  return _mm256_mask_cvtsepi64_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm_cvtsepi64_epi32(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.128
  return _mm_cvtsepi64_epi32(__A); 
}

__m128i test_mm_mask_cvtsepi64_epi32(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.128
  return _mm_mask_cvtsepi64_epi32(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtsepi64_epi32(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.128
  return _mm_maskz_cvtsepi64_epi32(__M, __A); 
}

void test_mm_mask_cvtsepi64_storeu_epi32(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.mem.128
  return _mm_mask_cvtsepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm256_cvtsepi64_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.256
  return _mm256_cvtsepi64_epi32(__A); 
}

__m128i test_mm256_mask_cvtsepi64_epi32(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.256
  return _mm256_mask_cvtsepi64_epi32(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtsepi64_epi32(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.256
  return _mm256_maskz_cvtsepi64_epi32(__M, __A); 
}

void test_mm256_mask_cvtsepi64_storeu_epi32(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.mem.256
  return _mm256_mask_cvtsepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm_cvtsepi64_epi16(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.128
  return _mm_cvtsepi64_epi16(__A); 
}

__m128i test_mm_mask_cvtsepi64_epi16(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.128
  return _mm_mask_cvtsepi64_epi16(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtsepi64_epi16(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.128
  return _mm_maskz_cvtsepi64_epi16(__M, __A); 
}

void test_mm_mask_cvtsepi64_storeu_epi16(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.mem.128
  return _mm_mask_cvtsepi64_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm256_cvtsepi64_epi16(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.256
  return _mm256_cvtsepi64_epi16(__A); 
}

__m128i test_mm256_mask_cvtsepi64_epi16(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.256
  return _mm256_mask_cvtsepi64_epi16(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtsepi64_epi16(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.256
  return _mm256_maskz_cvtsepi64_epi16(__M, __A); 
}

void test_mm256_mask_cvtsepi64_storeu_epi16(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.mem.256
  return _mm256_mask_cvtsepi64_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm_cvtusepi32_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.128
  return _mm_cvtusepi32_epi8(__A); 
}

__m128i test_mm_mask_cvtusepi32_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.128
  return _mm_mask_cvtusepi32_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtusepi32_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.128
  return _mm_maskz_cvtusepi32_epi8(__M, __A); 
}

void test_mm_mask_cvtusepi32_storeu_epi8(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.mem.128
  return _mm_mask_cvtusepi32_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm256_cvtusepi32_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.256
  return _mm256_cvtusepi32_epi8(__A); 
}

__m128i test_mm256_mask_cvtusepi32_epi8(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.256
  return _mm256_mask_cvtusepi32_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtusepi32_epi8(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.256
  return _mm256_maskz_cvtusepi32_epi8(__M, __A); 
}

void test_mm256_mask_cvtusepi32_storeu_epi8(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.mem.256
  return _mm256_mask_cvtusepi32_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm_cvtusepi32_epi16(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.128
  return _mm_cvtusepi32_epi16(__A); 
}

__m128i test_mm_mask_cvtusepi32_epi16(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.128
  return _mm_mask_cvtusepi32_epi16(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtusepi32_epi16(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.128
  return _mm_maskz_cvtusepi32_epi16(__M, __A); 
}

void test_mm_mask_cvtusepi32_storeu_epi16(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.mem.128
  return _mm_mask_cvtusepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm256_cvtusepi32_epi16(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.256
  return _mm256_cvtusepi32_epi16(__A); 
}

__m128i test_mm256_mask_cvtusepi32_epi16(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.256
  return _mm256_mask_cvtusepi32_epi16(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtusepi32_epi16(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.256
  return _mm256_maskz_cvtusepi32_epi16(__M, __A); 
}

void test_mm256_mask_cvtusepi32_storeu_epi16(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.mem.256
  return _mm256_mask_cvtusepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm_cvtusepi64_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.128
  return _mm_cvtusepi64_epi8(__A); 
}

__m128i test_mm_mask_cvtusepi64_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.128
  return _mm_mask_cvtusepi64_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtusepi64_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.128
  return _mm_maskz_cvtusepi64_epi8(__M, __A); 
}

void test_mm_mask_cvtusepi64_storeu_epi8(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.mem.128
  return _mm_mask_cvtusepi64_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm256_cvtusepi64_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.256
  return _mm256_cvtusepi64_epi8(__A); 
}

__m128i test_mm256_mask_cvtusepi64_epi8(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.256
  return _mm256_mask_cvtusepi64_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtusepi64_epi8(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.256
  return _mm256_maskz_cvtusepi64_epi8(__M, __A); 
}

void test_mm256_mask_cvtusepi64_storeu_epi8(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.mem.256
  return _mm256_mask_cvtusepi64_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm_cvtusepi64_epi32(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.128
  return _mm_cvtusepi64_epi32(__A); 
}

__m128i test_mm_mask_cvtusepi64_epi32(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.128
  return _mm_mask_cvtusepi64_epi32(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtusepi64_epi32(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.128
  return _mm_maskz_cvtusepi64_epi32(__M, __A); 
}

void test_mm_mask_cvtusepi64_storeu_epi32(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.mem.128
  return _mm_mask_cvtusepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm256_cvtusepi64_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.256
  return _mm256_cvtusepi64_epi32(__A); 
}

__m128i test_mm256_mask_cvtusepi64_epi32(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.256
  return _mm256_mask_cvtusepi64_epi32(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtusepi64_epi32(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.256
  return _mm256_maskz_cvtusepi64_epi32(__M, __A); 
}

void test_mm256_mask_cvtusepi64_storeu_epi32(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.mem.256
  return _mm256_mask_cvtusepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm_cvtusepi64_epi16(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.128
  return _mm_cvtusepi64_epi16(__A); 
}

__m128i test_mm_mask_cvtusepi64_epi16(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.128
  return _mm_mask_cvtusepi64_epi16(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtusepi64_epi16(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.128
  return _mm_maskz_cvtusepi64_epi16(__M, __A); 
}

void test_mm_mask_cvtusepi64_storeu_epi16(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.mem.128
  return _mm_mask_cvtusepi64_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm256_cvtusepi64_epi16(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.256
  return _mm256_cvtusepi64_epi16(__A); 
}

__m128i test_mm256_mask_cvtusepi64_epi16(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.256
  return _mm256_mask_cvtusepi64_epi16(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtusepi64_epi16(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.256
  return _mm256_maskz_cvtusepi64_epi16(__M, __A); 
}

void test_mm256_mask_cvtusepi64_storeu_epi16(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.mem.256
  return _mm256_mask_cvtusepi64_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm_cvtepi32_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepi32_epi8
  // CHECK: trunc <4 x i32> %{{.*}} to <4 x i8>
  // CHECK: shufflevector <4 x i8> %{{.*}}, <4 x i8> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  return _mm_cvtepi32_epi8(__A); 
}

__m128i test_mm_mask_cvtepi32_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.128
  return _mm_mask_cvtepi32_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtepi32_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.128
  return _mm_maskz_cvtepi32_epi8(__M, __A); 
}

void test_mm_mask_cvtepi32_storeu_epi8(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.mem.128
  return _mm_mask_cvtepi32_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm256_cvtepi32_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtepi32_epi8
  // CHECK: trunc <8 x i32> %{{.*}} to <8 x i8>
  // CHECK: shufflevector <8 x i8> %{{.*}}, <8 x i8> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm256_cvtepi32_epi8(__A); 
}

__m128i test_mm256_mask_cvtepi32_epi8(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.256
  return _mm256_mask_cvtepi32_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtepi32_epi8(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.256
  return _mm256_maskz_cvtepi32_epi8(__M, __A); 
}

void test_mm256_mask_cvtepi32_storeu_epi8(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.mem.256
  return _mm256_mask_cvtepi32_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm_cvtepi32_epi16(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepi32_epi16
  // CHECK: trunc <4 x i32> %{{.*}} to <4 x i16>
  // CHECK: shufflevector <4 x i16> %{{.*}}, <4 x i16> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm_cvtepi32_epi16(__A); 
}

__m128i test_mm_mask_cvtepi32_epi16(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.128
  return _mm_mask_cvtepi32_epi16(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtepi32_epi16(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.128
  return _mm_maskz_cvtepi32_epi16(__M, __A); 
}

void test_mm_mask_cvtepi32_storeu_epi16(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.mem.128
  return _mm_mask_cvtepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm256_cvtepi32_epi16(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtepi32_epi16
  // CHECK: trunc <8 x i32> %{{.*}} to <8 x i16>
  return _mm256_cvtepi32_epi16(__A); 
}

__m128i test_mm256_mask_cvtepi32_epi16(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.256
  return _mm256_mask_cvtepi32_epi16(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtepi32_epi16(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.256
  return _mm256_maskz_cvtepi32_epi16(__M, __A); 
}

void test_mm256_mask_cvtepi32_storeu_epi16(void *  __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.mem.256
  return _mm256_mask_cvtepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm_cvtepi64_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepi64_epi8
  // CHECK: trunc <2 x i64> %{{.*}} to <2 x i8>
  // CHECK: shufflevector <2 x i8> %{{.*}}, <2 x i8> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  return _mm_cvtepi64_epi8(__A); 
}

__m128i test_mm_mask_cvtepi64_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.128
  return _mm_mask_cvtepi64_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtepi64_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.128
  return _mm_maskz_cvtepi64_epi8(__M, __A); 
}

void test_mm_mask_cvtepi64_storeu_epi8(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.mem.128
  return _mm_mask_cvtepi64_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm256_cvtepi64_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtepi64_epi8
  // CHECK: trunc <4 x i64> %{{.*}} to <4 x i8>
  // CHECK: shufflevector <4 x i8> %{{.*}}, <4 x i8> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  return _mm256_cvtepi64_epi8(__A); 
}

__m128i test_mm256_mask_cvtepi64_epi8(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.256
  return _mm256_mask_cvtepi64_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtepi64_epi8(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.256
  return _mm256_maskz_cvtepi64_epi8(__M, __A); 
}

void test_mm256_mask_cvtepi64_storeu_epi8(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.mem.256
  return _mm256_mask_cvtepi64_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm_cvtepi64_epi32(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepi64_epi32
  // CHECK: trunc <2 x i64> %{{.*}} to <2 x i32>
  // CHECK: shufflevector <2 x i32> %{{.*}}, <2 x i32> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  return _mm_cvtepi64_epi32(__A); 
}

__m128i test_mm_mask_cvtepi64_epi32(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmov.qd.128
  return _mm_mask_cvtepi64_epi32(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtepi64_epi32(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmov.qd.128
  return _mm_maskz_cvtepi64_epi32(__M, __A); 
}

void test_mm_mask_cvtepi64_storeu_epi32(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmov.qd.mem.128
  return _mm_mask_cvtepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm256_cvtepi64_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtepi64_epi32
  // CHECK: trunc <4 x i64> %{{.*}} to <4 x i32>
  return _mm256_cvtepi64_epi32(__A); 
}

__m128i test_mm256_mask_cvtepi64_epi32(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi64_epi32
  // CHECK: trunc <4 x i64> %{{.*}} to <4 x i32>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm256_mask_cvtepi64_epi32(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtepi64_epi32(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi64_epi32
  // CHECK: trunc <4 x i64> %{{.*}} to <4 x i32>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm256_maskz_cvtepi64_epi32(__M, __A); 
}

void test_mm256_mask_cvtepi64_storeu_epi32(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmov.qd.mem.256
  return _mm256_mask_cvtepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm_cvtepi64_epi16(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepi64_epi16
  // CHECK: trunc <2 x i64> %{{.*}} to <2 x i16>
  // CHECK: shufflevector <2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 3, i32 3, i32 3, i32 3>
  return _mm_cvtepi64_epi16(__A); 
}

__m128i test_mm_mask_cvtepi64_epi16(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.128
  return _mm_mask_cvtepi64_epi16(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtepi64_epi16(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.128
  return _mm_maskz_cvtepi64_epi16(__M, __A); 
}

void test_mm_mask_cvtepi64_storeu_epi16(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.mem.128
  return _mm_mask_cvtepi64_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm256_cvtepi64_epi16(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtepi64_epi16
  // CHECK: trunc <4 x i64> %{{.*}} to <4 x i16>
  // CHECK: shufflevector <4 x i16> %{{.*}}, <4 x i16> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm256_cvtepi64_epi16(__A); 
}

__m128i test_mm256_mask_cvtepi64_epi16(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.256
  return _mm256_mask_cvtepi64_epi16(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtepi64_epi16(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.256
  return _mm256_maskz_cvtepi64_epi16(__M, __A); 
}

void test_mm256_mask_cvtepi64_storeu_epi16(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.mem.256
  return _mm256_mask_cvtepi64_storeu_epi16(__P, __M, __A); 
}

__m128 test_mm256_extractf32x4_ps(__m256 __A) {
  // CHECK-LABEL: @test_mm256_extractf32x4_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  return _mm256_extractf32x4_ps(__A, 1); 
}

__m128 test_mm256_mask_extractf32x4_ps(__m128 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_extractf32x4_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm256_mask_extractf32x4_ps(__W, __U, __A, 1); 
}

__m128 test_mm256_maskz_extractf32x4_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_extractf32x4_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm256_maskz_extractf32x4_ps(__U, __A, 1); 
}

__m128i test_mm256_extracti32x4_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm256_extracti32x4_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  return _mm256_extracti32x4_epi32(__A, 1); 
}

__m128i test_mm256_mask_extracti32x4_epi32(__m128i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_extracti32x4_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm256_mask_extracti32x4_epi32(__W, __U, __A, 1); 
}

__m128i test_mm256_maskz_extracti32x4_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_extracti32x4_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm256_maskz_extracti32x4_epi32(__U, __A, 1); 
}

__m256 test_mm256_insertf32x4(__m256 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm256_insertf32x4
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm256_insertf32x4(__A, __B, 1); 
}

__m256 test_mm256_mask_insertf32x4(__m256 __W, __mmask8 __U, __m256 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm256_mask_insertf32x4
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_insertf32x4(__W, __U, __A, __B, 1); 
}

__m256 test_mm256_maskz_insertf32x4(__mmask8 __U, __m256 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm256_maskz_insertf32x4
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_insertf32x4(__U, __A, __B, 1); 
}

__m256i test_mm256_inserti32x4(__m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_inserti32x4
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm256_inserti32x4(__A, __B, 1); 
}

__m256i test_mm256_mask_inserti32x4(__m256i __W, __mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_mask_inserti32x4
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_inserti32x4(__W, __U, __A, __B, 1); 
}

__m256i test_mm256_maskz_inserti32x4(__mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_inserti32x4
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_inserti32x4(__U, __A, __B, 1); 
}

__m128d test_mm_getmant_pd(__m128d __A) {
  // CHECK-LABEL: @test_mm_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.128
  return _mm_getmant_pd(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m128d test_mm_mask_getmant_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.128
  return _mm_mask_getmant_pd(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m128d test_mm_maskz_getmant_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.128
  return _mm_maskz_getmant_pd(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m256d test_mm256_getmant_pd(__m256d __A) {
  // CHECK-LABEL: @test_mm256_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.256
  return _mm256_getmant_pd(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m256d test_mm256_mask_getmant_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.256
  return _mm256_mask_getmant_pd(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m256d test_mm256_maskz_getmant_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.256
  return _mm256_maskz_getmant_pd(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m128 test_mm_getmant_ps(__m128 __A) {
  // CHECK-LABEL: @test_mm_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.128
  return _mm_getmant_ps(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m128 test_mm_mask_getmant_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.128
  return _mm_mask_getmant_ps(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m128 test_mm_maskz_getmant_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.128
  return _mm_maskz_getmant_ps(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m256 test_mm256_getmant_ps(__m256 __A) {
  // CHECK-LABEL: @test_mm256_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.256
  return _mm256_getmant_ps(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m256 test_mm256_mask_getmant_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.256
  return _mm256_mask_getmant_ps(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m256 test_mm256_maskz_getmant_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.256
  return _mm256_maskz_getmant_ps(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m128d test_mm_mmask_i64gather_pd(__m128d __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm_mmask_i64gather_pd
  // CHECK: @llvm.x86.avx512.mask.gather3div2.df
  return _mm_mmask_i64gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm_mmask_i64gather_epi64(__m128i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm_mmask_i64gather_epi64
  // CHECK: @llvm.x86.avx512.mask.gather3div2.di
  return _mm_mmask_i64gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m256d test_mm256_mmask_i64gather_pd(__m256d __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm256_mmask_i64gather_pd
  // CHECK: @llvm.x86.avx512.mask.gather3div4.df
  return _mm256_mmask_i64gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m256i test_mm256_mmask_i64gather_epi64(__m256i __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm256_mmask_i64gather_epi64
  // CHECK: @llvm.x86.avx512.mask.gather3div4.di
  return _mm256_mmask_i64gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m128 test_mm_mmask_i64gather_ps(__m128 __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm_mmask_i64gather_ps
  // CHECK: @llvm.x86.avx512.mask.gather3div4.sf
  return _mm_mmask_i64gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm_mmask_i64gather_epi32(__m128i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm_mmask_i64gather_epi32
  // CHECK: @llvm.x86.avx512.mask.gather3div4.si
  return _mm_mmask_i64gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m128 test_mm256_mmask_i64gather_ps(__m128 __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm256_mmask_i64gather_ps
  // CHECK: @llvm.x86.avx512.mask.gather3div8.sf
  return _mm256_mmask_i64gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm256_mmask_i64gather_epi32(__m128i __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm256_mmask_i64gather_epi32
  // CHECK: @llvm.x86.avx512.mask.gather3div8.si
  return _mm256_mmask_i64gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m128d test_mm_mask_i32gather_pd(__m128d __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm_mask_i32gather_pd
  // CHECK: @llvm.x86.avx512.mask.gather3siv2.df
  return _mm_mmask_i32gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm_mask_i32gather_epi64(__m128i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm_mask_i32gather_epi64
  // CHECK: @llvm.x86.avx512.mask.gather3siv2.di
  return _mm_mmask_i32gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m256d test_mm256_mask_i32gather_pd(__m256d __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm256_mask_i32gather_pd
  // CHECK: @llvm.x86.avx512.mask.gather3siv4.df
  return _mm256_mmask_i32gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m256i test_mm256_mask_i32gather_epi64(__m256i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm256_mask_i32gather_epi64
  // CHECK: @llvm.x86.avx512.mask.gather3siv4.di
  return _mm256_mmask_i32gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m128 test_mm_mask_i32gather_ps(__m128 __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm_mask_i32gather_ps
  // CHECK: @llvm.x86.avx512.mask.gather3siv4.sf
  return _mm_mmask_i32gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm_mask_i32gather_epi32(__m128i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm_mask_i32gather_epi32
  // CHECK: @llvm.x86.avx512.mask.gather3siv4.si
  return _mm_mmask_i32gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m256 test_mm256_mask_i32gather_ps(__m256 __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm256_mask_i32gather_ps
  // CHECK: @llvm.x86.avx512.mask.gather3siv8.sf
  return _mm256_mmask_i32gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m256i test_mm256_mask_i32gather_epi32(__m256i __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm256_mask_i32gather_epi32
  // CHECK: @llvm.x86.avx512.mask.gather3siv8.si
  return _mm256_mmask_i32gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m256d test_mm256_permutex_pd(__m256d __X) {
  // CHECK-LABEL: @test_mm256_permutex_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> poison, <4 x i32> <i32 3, i32 0, i32 0, i32 0>
  return _mm256_permutex_pd(__X, 3);
}

__m256d test_mm256_mask_permutex_pd(__m256d __W, __mmask8 __U, __m256d __X) {
  // CHECK-LABEL: @test_mm256_mask_permutex_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_permutex_pd(__W, __U, __X, 1);
}

__m256d test_mm256_maskz_permutex_pd(__mmask8 __U, __m256d __X) {
  // CHECK-LABEL: @test_mm256_maskz_permutex_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_permutex_pd(__U, __X, 1);
}

__m256i test_mm256_permutex_epi64(__m256i __X) {
  // CHECK-LABEL: @test_mm256_permutex_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> poison, <4 x i32> <i32 3, i32 0, i32 0, i32 0>
  return _mm256_permutex_epi64(__X, 3);
}

__m256i test_mm256_mask_permutex_epi64(__m256i __W, __mmask8 __M, __m256i __X) {
  // CHECK-LABEL: @test_mm256_mask_permutex_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> poison, <4 x i32> <i32 3, i32 0, i32 0, i32 0>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_permutex_epi64(__W, __M, __X, 3);
}

__m256i test_mm256_maskz_permutex_epi64(__mmask8 __M, __m256i __X) {
  // CHECK-LABEL: @test_mm256_maskz_permutex_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> poison, <4 x i32> <i32 3, i32 0, i32 0, i32 0>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_permutex_epi64(__M, __X, 3);
}

__m256d test_mm256_permutexvar_pd(__m256i __X, __m256d __Y) {
  // CHECK-LABEL: @test_mm256_permutexvar_pd
  // CHECK: @llvm.x86.avx512.permvar.df.256
  return _mm256_permutexvar_pd(__X, __Y);
}

__m256d test_mm256_mask_permutexvar_pd(__m256d __W, __mmask8 __U, __m256i __X, __m256d __Y) {
  // CHECK-LABEL: @test_mm256_mask_permutexvar_pd
  // CHECK: @llvm.x86.avx512.permvar.df.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_permutexvar_pd(__W, __U, __X, __Y);
}

__m256d test_mm256_maskz_permutexvar_pd(__mmask8 __U, __m256i __X, __m256d __Y) {
  // CHECK-LABEL: @test_mm256_maskz_permutexvar_pd
  // CHECK: @llvm.x86.avx512.permvar.df.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_permutexvar_pd(__U, __X, __Y);
}

__m256i test_mm256_maskz_permutexvar_epi64(__mmask8 __M, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_permutexvar_epi64
  // CHECK: @llvm.x86.avx512.permvar.di.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_permutexvar_epi64(__M, __X, __Y);
}

__m256i test_mm256_mask_permutexvar_epi64(__m256i __W, __mmask8 __M, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_permutexvar_epi64
  // CHECK: @llvm.x86.avx512.permvar.di.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_permutexvar_epi64(__W, __M, __X, __Y);
}

__m256 test_mm256_mask_permutexvar_ps(__m256 __W, __mmask8 __U, __m256i __X, __m256 __Y) {
  // CHECK-LABEL: @test_mm256_mask_permutexvar_ps
  // CHECK: @llvm.x86.avx2.permps
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_permutexvar_ps(__W, __U, __X, __Y);
}

__m256 test_mm256_maskz_permutexvar_ps(__mmask8 __U, __m256i __X, __m256 __Y) {
  // CHECK-LABEL: @test_mm256_maskz_permutexvar_ps
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_permutexvar_ps(__U, __X, __Y);
}

__m256 test_mm256_permutexvar_ps(__m256i __X, __m256 __Y) {
  // CHECK-LABEL: @test_mm256_permutexvar_ps
  // CHECK: @llvm.x86.avx2.permps
  return _mm256_permutexvar_ps( __X, __Y);
}

__m256i test_mm256_maskz_permutexvar_epi32(__mmask8 __M, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_permutexvar_epi32
  // CHECK: @llvm.x86.avx2.permd
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_permutexvar_epi32(__M, __X, __Y);
}

__m256i test_mm256_permutexvar_epi32(__m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_permutexvar_epi32
  // CHECK: @llvm.x86.avx2.permd
  return _mm256_permutexvar_epi32(__X, __Y);
}

__m256i test_mm256_mask_permutexvar_epi32(__m256i __W, __mmask8 __M, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_permutexvar_epi32
  // CHECK: @llvm.x86.avx2.permd
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_permutexvar_epi32(__W, __M, __X, __Y);
}

__m128i test_mm_alignr_epi32(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_alignr_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  return _mm_alignr_epi32(__A, __B, 1);
}

__m128i test_mm_mask_alignr_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_alignr_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_alignr_epi32(__W, __U, __A, __B, 5);
}

__m128i test_mm_maskz_alignr_epi32(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_alignr_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_alignr_epi32(__U, __A, __B, 1);
}

__m256i test_mm256_alignr_epi32(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_alignr_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  return _mm256_alignr_epi32(__A, __B, 1);
}

__m256i test_mm256_mask_alignr_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_alignr_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_alignr_epi32(__W, __U, __A, __B, 9);
}

__m256i test_mm256_maskz_alignr_epi32(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_alignr_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_alignr_epi32(__U, __A, __B, 1);
}

__m128i test_mm_alignr_epi64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_alignr_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 2>
  return _mm_alignr_epi64(__A, __B, 1);
}

__m128i test_mm_mask_alignr_epi64(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_alignr_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 2>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_alignr_epi64(__W, __U, __A, __B, 3);
}

__m128i test_mm_maskz_alignr_epi64(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_alignr_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 2>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_alignr_epi64(__U, __A, __B, 1);
}

__m256i test_mm256_alignr_epi64(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_alignr_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  return _mm256_alignr_epi64(__A, __B, 1);
}

__m256i test_mm256_mask_alignr_epi64(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_alignr_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_alignr_epi64(__W, __U, __A, __B, 5);
}

__m256i test_mm256_maskz_alignr_epi64(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_alignr_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_alignr_epi64(__U, __A, __B, 1);
}

__m128 test_mm_mask_movehdup_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_movehdup_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 1, i32 1, i32 3, i32 3>
  // CHECK: select <4 x i1> %{{.*}} <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_movehdup_ps(__W, __U, __A);
}

__m128 test_mm_maskz_movehdup_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_movehdup_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 1, i32 1, i32 3, i32 3>
  // CHECK: select <4 x i1> %{{.*}} <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_movehdup_ps(__U, __A);
}

__m256 test_mm256_mask_movehdup_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_movehdup_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  // CHECK: select <8 x i1> %{{.*}} <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_movehdup_ps(__W, __U, __A);
}

__m256 test_mm256_maskz_movehdup_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_movehdup_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  // CHECK: select <8 x i1> %{{.*}} <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_movehdup_ps(__U, __A);
}

__m128 test_mm_mask_moveldup_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_moveldup_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  // CHECK: select <4 x i1> %{{.*}} <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_moveldup_ps(__W, __U, __A);
}

__m128 test_mm_maskz_moveldup_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_moveldup_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  // CHECK: select <4 x i1> %{{.*}} <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_moveldup_ps(__U, __A);
}

__m256 test_mm256_mask_moveldup_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_moveldup_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  // CHECK: select <8 x i1> %{{.*}} <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_moveldup_ps(__W, __U, __A);
}

__m256 test_mm256_maskz_moveldup_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_moveldup_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  // CHECK: select <8 x i1> %{{.*}} <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_moveldup_ps(__U, __A);
}

__m128i test_mm_mask_shuffle_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_shuffle_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_shuffle_epi32(__W, __U, __A, 1);
}

__m128i test_mm_maskz_shuffle_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_shuffle_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <4 x i32> <i32 2, i32 0, i32 0, i32 0>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_shuffle_epi32(__U, __A, 2);
}

__m256i test_mm256_mask_shuffle_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_shuffle_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> poison, <8 x i32> <i32 2, i32 0, i32 0, i32 0, i32 6, i32 4, i32 4, i32 4>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_shuffle_epi32(__W, __U, __A, 2);
}

__m256i test_mm256_maskz_shuffle_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_shuffle_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> poison, <8 x i32> <i32 2, i32 0, i32 0, i32 0, i32 6, i32 4, i32 4, i32 4>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_shuffle_epi32(__U, __A, 2);
}

__m128d test_mm_mask_mov_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_mov_pd
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_mov_pd(__W, __U, __A);
}

__m128d test_mm_maskz_mov_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_mov_pd
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_mov_pd(__U, __A);
}

__m256d test_mm256_mask_mov_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_mov_pd
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_mov_pd(__W, __U, __A);
}

__m256d test_mm256_maskz_mov_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_mov_pd
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_mov_pd(__U, __A);
}

__m128 test_mm_mask_mov_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_mov_ps
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_mov_ps(__W, __U, __A);
}

__m128 test_mm_maskz_mov_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_mov_ps
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_mov_ps(__U, __A);
}

__m256 test_mm256_mask_mov_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_mov_ps
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_mov_ps(__W, __U, __A);
}

__m256 test_mm256_maskz_mov_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_mov_ps
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_mov_ps(__U, __A);
}

__m128 test_mm_mask_cvtph_ps(__m128 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtph_ps
  // CHECK: bitcast <2 x i64> %{{.*}} to <8 x i16>
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: bitcast <4 x i16> %{{.*}} to <4 x half>
  // CHECK: fpext <4 x half> %{{.*}} to <4 x float>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_cvtph_ps(__W, __U, __A);
}

__m128 test_mm_maskz_cvtph_ps(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtph_ps
  // CHECK: bitcast <2 x i64> %{{.*}} to <8 x i16>
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: bitcast <4 x i16> %{{.*}} to <4 x half>
  // CHECK: fpext <4 x half> %{{.*}} to <4 x float>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_cvtph_ps(__U, __A);
}

__m256 test_mm256_mask_cvtph_ps(__m256 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtph_ps
  // CHECK: bitcast <2 x i64> %{{.*}} to <8 x i16>
  // CHECK: bitcast <8 x i16> %{{.*}} to <8 x half>
  // CHECK: fpext <8 x half> %{{.*}} to <8 x float>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_cvtph_ps(__W, __U, __A);
}

__m256 test_mm256_maskz_cvtph_ps(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtph_ps
  // CHECK: bitcast <2 x i64> %{{.*}} to <8 x i16>
  // CHECK: bitcast <8 x i16> %{{.*}} to <8 x half>
  // CHECK: fpext <8 x half> %{{.*}} to <8 x float>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_cvtph_ps(__U, __A);
}

__m128i test_mm_mask_cvtps_ph(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvtps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.128
  return _mm_mask_cvtps_ph(__W, __U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm_maskz_cvtps_ph(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.128
  return _mm_maskz_cvtps_ph(__U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_mask_cvtps_ph(__m128i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.256
  return _mm256_mask_cvtps_ph(__W, __U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_maskz_cvtps_ph(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.256
  return _mm256_maskz_cvtps_ph(__U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm_mask_cvt_roundps_ph(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvt_roundps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.128
  return _mm_mask_cvt_roundps_ph(__W, __U, __A, _MM_FROUND_TO_ZERO);
}

__m128i test_mm_maskz_cvt_roundps_ph(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvt_roundps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.128
  return _mm_maskz_cvt_roundps_ph(__U, __A, _MM_FROUND_TO_ZERO);
}

__m128i test_mm256_mask_cvt_roundps_ph(__m128i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvt_roundps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.256
  return _mm256_mask_cvt_roundps_ph(__W, __U, __A, _MM_FROUND_TO_ZERO);
}

__m128i test_mm256_maskz_cvt_roundps_ph(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvt_roundps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.256
  return _mm256_maskz_cvt_roundps_ph(__U, __A, _MM_FROUND_TO_ZERO);
}

__mmask8 test_mm_cmpeq_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epi32_mask
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpeq_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epi32_mask
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpeq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_mask_cmpeq_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epi64_mask
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpeq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epi64_mask
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpeq_epi64_mask(__a, __b);
}

__mmask8 test_mm_cmpgt_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epi32_mask
  // CHECK: icmp sgt <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpgt_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epi32_mask
  // CHECK: icmp sgt <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpgt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_mask_cmpgt_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epi64_mask
  // CHECK: icmp sgt <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpgt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epi64_mask
  // CHECK: icmp sgt <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpgt_epi64_mask(__a, __b);
}

__mmask8 test_mm256_cmpeq_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epi32_mask
  // CHECK: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpeq_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpeq_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epi32_mask
  // CHECK: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpeq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_mask_cmpeq_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epi64_mask
  // CHECK: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpeq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpeq_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epi64_mask
  // CHECK: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpeq_epi64_mask(__a, __b);
}

__mmask8 test_mm256_cmpgt_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epi32_mask
  // CHECK: icmp sgt <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpgt_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epi32_mask
  // CHECK: icmp sgt <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpgt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epi64_mask
  // CHECK: icmp sgt <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpgt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpgt_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epi64_mask
  // CHECK: icmp sgt <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpgt_epi64_mask(__a, __b);
}
