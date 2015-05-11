// RUN: %clang_cc1 %s -O0 -triple=x86_64-apple-darwin -ffreestanding -target-feature +avx512bw -emit-llvm -o - -Werror | FileCheck %s

#include <immintrin.h>

__mmask64 test_mm512_cmpeq_epi8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpeq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.b.512
  return (__mmask64)_mm512_cmpeq_epi8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmpeq_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpeq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.b.512
  return (__mmask64)_mm512_mask_cmpeq_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmpeq_epi16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpeq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.w.512
  return (__mmask32)_mm512_cmpeq_epi16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmpeq_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpeq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.w.512
  return (__mmask32)_mm512_mask_cmpeq_epi16_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmpgt_epi8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpgt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.b.512
  return (__mmask64)_mm512_cmpgt_epi8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmpgt_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpgt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.b.512
  return (__mmask64)_mm512_mask_cmpgt_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmpgt_epi16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpgt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.w.512
  return (__mmask32)_mm512_cmpgt_epi16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmpgt_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpgt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.w.512
  return (__mmask32)_mm512_mask_cmpgt_epi16_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmpeq_epu8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpeq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 0, i64 -1)
  return (__mmask64)_mm512_cmpeq_epu8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmpeq_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpeq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 0, i64 {{.*}})
  return (__mmask64)_mm512_mask_cmpeq_epu8_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmpeq_epu16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpeq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 0, i32 -1)
  return (__mmask32)_mm512_cmpeq_epu16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmpeq_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpeq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 0, i32 {{.*}})
  return (__mmask32)_mm512_mask_cmpeq_epu16_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmpgt_epu8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpgt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 6, i64 -1)
  return (__mmask64)_mm512_cmpgt_epu8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmpgt_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpgt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 6, i64 {{.*}})
  return (__mmask64)_mm512_mask_cmpgt_epu8_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmpgt_epu16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpgt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 6, i32 -1)
  return (__mmask32)_mm512_cmpgt_epu16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmpgt_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpgt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 6, i32 {{.*}})
  return (__mmask32)_mm512_mask_cmpgt_epu16_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmpge_epi8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpge_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 5, i64 -1)
  return (__mmask64)_mm512_cmpge_epi8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmpge_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpge_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 5, i64 {{.*}})
  return (__mmask64)_mm512_mask_cmpge_epi8_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmpge_epu8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpge_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 5, i64 -1)
  return (__mmask64)_mm512_cmpge_epu8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmpge_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpge_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 5, i64 {{.*}})
  return (__mmask64)_mm512_mask_cmpge_epu8_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmpge_epi16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpge_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 5, i32 -1)
  return (__mmask32)_mm512_cmpge_epi16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmpge_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpge_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 5, i32 {{.*}})
  return (__mmask32)_mm512_mask_cmpge_epi16_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmpge_epu16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpge_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 5, i32 -1)
  return (__mmask32)_mm512_cmpge_epu16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmpge_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpge_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 5, i32 {{.*}})
  return (__mmask32)_mm512_mask_cmpge_epu16_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmple_epi8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmple_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 2, i64 -1)
  return (__mmask64)_mm512_cmple_epi8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmple_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmple_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 2, i64 {{.*}})
  return (__mmask64)_mm512_mask_cmple_epi8_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmple_epu8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmple_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 2, i64 -1)
  return (__mmask64)_mm512_cmple_epu8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmple_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmple_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 2, i64 {{.*}})
  return (__mmask64)_mm512_mask_cmple_epu8_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmple_epi16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmple_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 2, i32 -1)
  return (__mmask32)_mm512_cmple_epi16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmple_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmple_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 2, i32 {{.*}})
  return (__mmask32)_mm512_mask_cmple_epi16_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmple_epu16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmple_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 2, i32 -1)
  return (__mmask32)_mm512_cmple_epu16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmple_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmple_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 2, i32 {{.*}})
  return (__mmask32)_mm512_mask_cmple_epu16_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmplt_epi8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmplt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 1, i64 -1)
  return (__mmask64)_mm512_cmplt_epi8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmplt_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmplt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 1, i64 {{.*}})
  return (__mmask64)_mm512_mask_cmplt_epi8_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmplt_epu8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmplt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 1, i64 -1)
  return (__mmask64)_mm512_cmplt_epu8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmplt_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmplt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 1, i64 {{.*}})
  return (__mmask64)_mm512_mask_cmplt_epu8_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmplt_epi16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmplt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 1, i32 -1)
  return (__mmask32)_mm512_cmplt_epi16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmplt_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmplt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 1, i32 {{.*}})
  return (__mmask32)_mm512_mask_cmplt_epi16_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmplt_epu16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmplt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 1, i32 -1)
  return (__mmask32)_mm512_cmplt_epu16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmplt_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmplt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 1, i32 {{.*}})
  return (__mmask32)_mm512_mask_cmplt_epu16_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmpneq_epi8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpneq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 4, i64 -1)
  return (__mmask64)_mm512_cmpneq_epi8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmpneq_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpneq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 4, i64 {{.*}})
  return (__mmask64)_mm512_mask_cmpneq_epi8_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmpneq_epu8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpneq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 4, i64 -1)
  return (__mmask64)_mm512_cmpneq_epu8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmpneq_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpneq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 4, i64 {{.*}})
  return (__mmask64)_mm512_mask_cmpneq_epu8_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmpneq_epi16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpneq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 4, i32 -1)
  return (__mmask32)_mm512_cmpneq_epi16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmpneq_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpneq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 4, i32 {{.*}})
  return (__mmask32)_mm512_mask_cmpneq_epi16_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmpneq_epu16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpneq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 4, i32 -1)
  return (__mmask32)_mm512_cmpneq_epu16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmpneq_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpneq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 4, i32 {{.*}})
  return (__mmask32)_mm512_mask_cmpneq_epu16_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmp_epi8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmp_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 7, i64 -1)
  return (__mmask64)_mm512_cmp_epi8_mask(__a, __b, 7);
}

__mmask64 test_mm512_mask_cmp_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 7, i64 {{.*}})
  return (__mmask64)_mm512_mask_cmp_epi8_mask(__u, __a, __b, 7);
}

__mmask64 test_mm512_cmp_epu8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmp_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 7, i64 -1)
  return (__mmask64)_mm512_cmp_epu8_mask(__a, __b, 7);
}

__mmask64 test_mm512_mask_cmp_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.512(<64 x i8> {{.*}}, <64 x i8> {{.*}}, i32 7, i64 {{.*}})
  return (__mmask64)_mm512_mask_cmp_epu8_mask(__u, __a, __b, 7);
}

__mmask32 test_mm512_cmp_epi16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmp_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 7, i32 -1)
  return (__mmask32)_mm512_cmp_epi16_mask(__a, __b, 7);
}

__mmask32 test_mm512_mask_cmp_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 7, i32 {{.*}})
  return (__mmask32)_mm512_mask_cmp_epi16_mask(__u, __a, __b, 7);
}

__mmask32 test_mm512_cmp_epu16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmp_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 7, i32 -1)
  return (__mmask32)_mm512_cmp_epu16_mask(__a, __b, 7);
}

__mmask32 test_mm512_mask_cmp_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.512(<32 x i16> {{.*}}, <32 x i16> {{.*}}, i32 7, i32 {{.*}})
  return (__mmask32)_mm512_mask_cmp_epu16_mask(__u, __a, __b, 7);
}

__m512i test_mm512_add_epi8 (__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_add_epi8
  //CHECK: add <64 x i8>
  return _mm512_add_epi8(__A,__B);
}

__m512i test_mm512_mask_add_epi8 (__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_mask_add_epi8
  //CHECK: @llvm.x86.avx512.mask.padd.b.512
  return _mm512_mask_add_epi8(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_add_epi8 (__mmask64 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_add_epi8
  //CHECK: @llvm.x86.avx512.mask.padd.b.512
  return _mm512_maskz_add_epi8(__U, __A, __B);
}

__m512i test_mm512_sub_epi8 (__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_sub_epi8
  //CHECK: sub <64 x i8>
  return _mm512_sub_epi8(__A, __B);
}

__m512i test_mm512_mask_sub_epi8 (__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_mask_sub_epi8
  //CHECK: @llvm.x86.avx512.mask.psub.b.512
  return _mm512_mask_sub_epi8(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_sub_epi8 (__mmask64 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_sub_epi8
  //CHECK: @llvm.x86.avx512.mask.psub.b.512
  return _mm512_maskz_sub_epi8(__U, __A, __B);
}

__m512i test_mm512_add_epi16 (__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_add_epi16
  //CHECK: add <32 x i16>
  return _mm512_add_epi16(__A, __B);
}

__m512i test_mm512_mask_add_epi16 (__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_mask_add_epi16
  //CHECK: @llvm.x86.avx512.mask.padd.w.512
  return _mm512_mask_add_epi16(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_add_epi16 (__mmask32 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_add_epi16
  //CHECK: @llvm.x86.avx512.mask.padd.w.512
  return _mm512_maskz_add_epi16(__U, __A, __B);
}

__m512i test_mm512_sub_epi16 (__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_sub_epi16
  //CHECK: sub <32 x i16>
  return _mm512_sub_epi16(__A, __B);
}

__m512i test_mm512_mask_sub_epi16 (__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_mask_sub_epi16
  //CHECK: @llvm.x86.avx512.mask.psub.w.512
  return _mm512_mask_sub_epi16(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_sub_epi16 (__mmask32 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_sub_epi16
  //CHECK: @llvm.x86.avx512.mask.psub.w.512
  return _mm512_maskz_sub_epi16(__U, __A, __B);
}

__m512i test_mm512_mullo_epi16 (__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_mullo_epi16
  //CHECK: mul <32 x i16>
  return _mm512_mullo_epi16(__A, __B);
}

__m512i test_mm512_mask_mullo_epi16 (__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_mask_mullo_epi16
  //CHECK: @llvm.x86.avx512.mask.pmull.w.512
  return _mm512_mask_mullo_epi16(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_mullo_epi16 (__mmask32 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_mullo_epi16
  //CHECK: @llvm.x86.avx512.mask.pmull.w.512
  return _mm512_maskz_mullo_epi16(__U, __A, __B);
}
