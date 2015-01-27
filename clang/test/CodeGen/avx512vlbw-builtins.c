// RUN: %clang_cc1 %s -O0 -triple=x86_64-apple-darwin -ffreestanding -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - -Werror | FileCheck %s

#include <immintrin.h>

__mmask32 test_mm256_cmpeq_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.b.256
  return (__mmask32)_mm256_cmpeq_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpeq_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.b.256
  return (__mmask32)_mm256_mask_cmpeq_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpeq_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.b.128
  return (__mmask16)_mm_cmpeq_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpeq_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.b.128
  return (__mmask16)_mm_mask_cmpeq_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpeq_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.w.256
  return (__mmask16)_mm256_cmpeq_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpeq_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.w.256
  return (__mmask16)_mm256_mask_cmpeq_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.w.128
  return (__mmask8)_mm_cmpeq_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.w.128
  return (__mmask8)_mm_mask_cmpeq_epi16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpgt_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.b.256
  return (__mmask32)_mm256_cmpgt_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpgt_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.b.256
  return (__mmask32)_mm256_mask_cmpgt_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpgt_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.b.128
  return (__mmask16)_mm_cmpgt_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpgt_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.b.128
  return (__mmask16)_mm_mask_cmpgt_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpgt_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.w.256
  return (__mmask16)_mm256_cmpgt_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpgt_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.w.256
  return (__mmask16)_mm256_mask_cmpgt_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.w.128
  return (__mmask8)_mm_cmpgt_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.w.128
  return (__mmask8)_mm_mask_cmpgt_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpeq_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 0, i16 -1)
  return (__mmask64)_mm_cmpeq_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpeq_epu8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 0, i16 {{.*}})
  return (__mmask64)_mm_mask_cmpeq_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 0, i8 -1)
  return (__mmask32)_mm_cmpeq_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epu16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 0, i8 {{.*}})
  return (__mmask32)_mm_mask_cmpeq_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpeq_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 0, i32 -1)
  return (__mmask64)_mm256_cmpeq_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpeq_epu8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 0, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmpeq_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpeq_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 0, i16 -1)
  return (__mmask32)_mm256_cmpeq_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpeq_epu16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 0, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmpeq_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpgt_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 6, i16 -1)
  return (__mmask64)_mm_cmpgt_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpgt_epu8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 6, i16 {{.*}})
  return (__mmask64)_mm_mask_cmpgt_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 6, i8 -1)
  return (__mmask32)_mm_cmpgt_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epu16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 6, i8 {{.*}})
  return (__mmask32)_mm_mask_cmpgt_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpgt_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 6, i32 -1)
  return (__mmask64)_mm256_cmpgt_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpgt_epu8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 6, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmpgt_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpgt_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 6, i16 -1)
  return (__mmask32)_mm256_cmpgt_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpgt_epu16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 6, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmpgt_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpge_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 5, i16 -1)
  return (__mmask64)_mm_cmpge_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpge_epi8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 5, i16 {{.*}})
  return (__mmask64)_mm_mask_cmpge_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpge_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 5, i16 -1)
  return (__mmask64)_mm_cmpge_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpge_epu8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 5, i16 {{.*}})
  return (__mmask64)_mm_mask_cmpge_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 5, i8 -1)
  return (__mmask32)_mm_cmpge_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epi16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 5, i8 {{.*}})
  return (__mmask32)_mm_mask_cmpge_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 5, i8 -1)
  return (__mmask32)_mm_cmpge_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epu16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 5, i8 {{.*}})
  return (__mmask32)_mm_mask_cmpge_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpge_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 5, i32 -1)
  return (__mmask64)_mm256_cmpge_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpge_epi8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 5, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmpge_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpge_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 5, i32 -1)
  return (__mmask64)_mm256_cmpge_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpge_epu8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 5, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmpge_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpge_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 5, i16 -1)
  return (__mmask32)_mm256_cmpge_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpge_epi16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 5, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmpge_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpge_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 5, i16 -1)
  return (__mmask32)_mm256_cmpge_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpge_epu16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 5, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmpge_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmple_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 2, i16 -1)
  return (__mmask64)_mm_cmple_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmple_epi8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 2, i16 {{.*}})
  return (__mmask64)_mm_mask_cmple_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmple_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 2, i16 -1)
  return (__mmask64)_mm_cmple_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmple_epu8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 2, i16 {{.*}})
  return (__mmask64)_mm_mask_cmple_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 2, i8 -1)
  return (__mmask32)_mm_cmple_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epi16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 2, i8 {{.*}})
  return (__mmask32)_mm_mask_cmple_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 2, i8 -1)
  return (__mmask32)_mm_cmple_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epu16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 2, i8 {{.*}})
  return (__mmask32)_mm_mask_cmple_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmple_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 2, i32 -1)
  return (__mmask64)_mm256_cmple_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmple_epi8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 2, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmple_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmple_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 2, i32 -1)
  return (__mmask64)_mm256_cmple_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmple_epu8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 2, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmple_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmple_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 2, i16 -1)
  return (__mmask32)_mm256_cmple_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmple_epi16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 2, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmple_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmple_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 2, i16 -1)
  return (__mmask32)_mm256_cmple_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmple_epu16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 2, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmple_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmplt_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 1, i16 -1)
  return (__mmask64)_mm_cmplt_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmplt_epi8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 1, i16 {{.*}})
  return (__mmask64)_mm_mask_cmplt_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmplt_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 1, i16 -1)
  return (__mmask64)_mm_cmplt_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmplt_epu8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 1, i16 {{.*}})
  return (__mmask64)_mm_mask_cmplt_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 1, i8 -1)
  return (__mmask32)_mm_cmplt_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epi16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 1, i8 {{.*}})
  return (__mmask32)_mm_mask_cmplt_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 1, i8 -1)
  return (__mmask32)_mm_cmplt_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epu16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 1, i8 {{.*}})
  return (__mmask32)_mm_mask_cmplt_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmplt_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 1, i32 -1)
  return (__mmask64)_mm256_cmplt_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmplt_epi8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 1, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmplt_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmplt_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 1, i32 -1)
  return (__mmask64)_mm256_cmplt_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmplt_epu8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 1, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmplt_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmplt_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 1, i16 -1)
  return (__mmask32)_mm256_cmplt_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmplt_epi16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 1, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmplt_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmplt_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 1, i16 -1)
  return (__mmask32)_mm256_cmplt_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmplt_epu16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 1, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmplt_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpneq_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 4, i16 -1)
  return (__mmask64)_mm_cmpneq_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpneq_epi8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 4, i16 {{.*}})
  return (__mmask64)_mm_mask_cmpneq_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpneq_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 4, i16 -1)
  return (__mmask64)_mm_cmpneq_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpneq_epu8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 4, i16 {{.*}})
  return (__mmask64)_mm_mask_cmpneq_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 4, i8 -1)
  return (__mmask32)_mm_cmpneq_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epi16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 4, i8 {{.*}})
  return (__mmask32)_mm_mask_cmpneq_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 4, i8 -1)
  return (__mmask32)_mm_cmpneq_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epu16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 4, i8 {{.*}})
  return (__mmask32)_mm_mask_cmpneq_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpneq_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 4, i32 -1)
  return (__mmask64)_mm256_cmpneq_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpneq_epi8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 4, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmpneq_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpneq_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 4, i32 -1)
  return (__mmask64)_mm256_cmpneq_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpneq_epu8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 4, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmpneq_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpneq_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 4, i16 -1)
  return (__mmask32)_mm256_cmpneq_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpneq_epi16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 4, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmpneq_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpneq_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 4, i16 -1)
  return (__mmask32)_mm256_cmpneq_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpneq_epu16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 4, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmpneq_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmp_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 7, i16 -1)
  return (__mmask64)_mm_cmp_epi8_mask(__a, __b, 7);
}

__mmask16 test_mm_mask_cmp_epi8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 7, i16 {{.*}})
  return (__mmask64)_mm_mask_cmp_epi8_mask(__u, __a, __b, 7);
}

__mmask16 test_mm_cmp_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 7, i16 -1)
  return (__mmask64)_mm_cmp_epu8_mask(__a, __b, 7);
}

__mmask16 test_mm_mask_cmp_epu8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i8 7, i16 {{.*}})
  return (__mmask64)_mm_mask_cmp_epu8_mask(__u, __a, __b, 7);
}

__mmask8 test_mm_cmp_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 7, i8 -1)
  return (__mmask32)_mm_cmp_epi16_mask(__a, __b, 7);
}

__mmask8 test_mm_mask_cmp_epi16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 7, i8 {{.*}})
  return (__mmask32)_mm_mask_cmp_epi16_mask(__u, __a, __b, 7);
}

__mmask8 test_mm_cmp_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 7, i8 -1)
  return (__mmask32)_mm_cmp_epu16_mask(__a, __b, 7);
}

__mmask8 test_mm_mask_cmp_epu16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i8 7, i8 {{.*}})
  return (__mmask32)_mm_mask_cmp_epu16_mask(__u, __a, __b, 7);
}

__mmask32 test_mm256_cmp_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 7, i32 -1)
  return (__mmask64)_mm256_cmp_epi8_mask(__a, __b, 7);
}

__mmask32 test_mm256_mask_cmp_epi8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 7, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmp_epi8_mask(__u, __a, __b, 7);
}

__mmask32 test_mm256_cmp_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 7, i32 -1)
  return (__mmask64)_mm256_cmp_epu8_mask(__a, __b, 7);
}

__mmask32 test_mm256_mask_cmp_epu8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i8 7, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmp_epu8_mask(__u, __a, __b, 7);
}

__mmask16 test_mm256_cmp_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 7, i16 -1)
  return (__mmask32)_mm256_cmp_epi16_mask(__a, __b, 7);
}

__mmask16 test_mm256_mask_cmp_epi16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 7, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmp_epi16_mask(__u, __a, __b, 7);
}

__mmask16 test_mm256_cmp_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 7, i16 -1)
  return (__mmask32)_mm256_cmp_epu16_mask(__a, __b, 7);
}

__mmask16 test_mm256_mask_cmp_epu16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i8 7, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmp_epu16_mask(__u, __a, __b, 7);
}
