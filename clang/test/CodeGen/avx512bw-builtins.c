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

__m512i test_mm512_mask_blend_epi8(__mmask64 __U, __m512i __A, __m512i __W) {
  // CHECK-LABEL: @test_mm512_mask_blend_epi8
  // CHECK: @llvm.x86.avx512.mask.blend.b.512
  return _mm512_mask_blend_epi8(__U,__A,__W); 
}
__m512i test_mm512_mask_blend_epi16(__mmask32 __U, __m512i __A, __m512i __W) {
  // CHECK-LABEL: @test_mm512_mask_blend_epi16
  // CHECK: @llvm.x86.avx512.mask.blend.w.512
  return _mm512_mask_blend_epi16(__U,__A,__W); 
}
__m512i test_mm512_abs_epi8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_abs_epi8
  // CHECK: @llvm.x86.avx512.mask.pabs.b.512
  return _mm512_abs_epi8(__A); 
}
__m512i test_mm512_mask_abs_epi8(__m512i __W, __mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_abs_epi8
  // CHECK: @llvm.x86.avx512.mask.pabs.b.512
  return _mm512_mask_abs_epi8(__W,__U,__A); 
}
__m512i test_mm512_maskz_abs_epi8(__mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_abs_epi8
  // CHECK: @llvm.x86.avx512.mask.pabs.b.512
  return _mm512_maskz_abs_epi8(__U,__A); 
}
__m512i test_mm512_abs_epi16(__m512i __A) {
  // CHECK-LABEL: @test_mm512_abs_epi16
  // CHECK: @llvm.x86.avx512.mask.pabs.w.512
  return _mm512_abs_epi16(__A); 
}
__m512i test_mm512_mask_abs_epi16(__m512i __W, __mmask32 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_abs_epi16
  // CHECK: @llvm.x86.avx512.mask.pabs.w.512
  return _mm512_mask_abs_epi16(__W,__U,__A); 
}
__m512i test_mm512_maskz_abs_epi16(__mmask32 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_abs_epi16
  // CHECK: @llvm.x86.avx512.mask.pabs.w.512
  return _mm512_maskz_abs_epi16(__U,__A); 
}
__m512i test_mm512_packs_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_packs_epi32
  // CHECK: @llvm.x86.avx512.mask.packssdw.512
  return _mm512_packs_epi32(__A,__B); 
}
__m512i test_mm512_maskz_packs_epi32(__mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_packs_epi32
  // CHECK: @llvm.x86.avx512.mask.packssdw.512
  return _mm512_maskz_packs_epi32(__M,__A,__B); 
}
__m512i test_mm512_mask_packs_epi32(__m512i __W, __mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_packs_epi32
  // CHECK: @llvm.x86.avx512.mask.packssdw.512
  return _mm512_mask_packs_epi32(__W,__M,__A,__B); 
}
__m512i test_mm512_packs_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_packs_epi16
  // CHECK: @llvm.x86.avx512.mask.packsswb.512
  return _mm512_packs_epi16(__A,__B); 
}
__m512i test_mm512_mask_packs_epi16(__m512i __W, __mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_packs_epi16
  // CHECK: @llvm.x86.avx512.mask.packsswb.512
  return _mm512_mask_packs_epi16(__W,__M,__A,__B); 
}
__m512i test_mm512_maskz_packs_epi16(__mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_packs_epi16
  // CHECK: @llvm.x86.avx512.mask.packsswb.512
  return _mm512_maskz_packs_epi16(__M,__A,__B); 
}
__m512i test_mm512_packus_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_packus_epi32
  // CHECK: @llvm.x86.avx512.mask.packusdw.512
  return _mm512_packus_epi32(__A,__B); 
}
__m512i test_mm512_maskz_packus_epi32(__mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_packus_epi32
  // CHECK: @llvm.x86.avx512.mask.packusdw.512
  return _mm512_maskz_packus_epi32(__M,__A,__B); 
}
__m512i test_mm512_mask_packus_epi32(__m512i __W, __mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_packus_epi32
  // CHECK: @llvm.x86.avx512.mask.packusdw.512
  return _mm512_mask_packus_epi32(__W,__M,__A,__B); 
}
__m512i test_mm512_packus_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_packus_epi16
  // CHECK: @llvm.x86.avx512.mask.packuswb.512
  return _mm512_packus_epi16(__A,__B); 
}
__m512i test_mm512_mask_packus_epi16(__m512i __W, __mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_packus_epi16
  // CHECK: @llvm.x86.avx512.mask.packuswb.512
  return _mm512_mask_packus_epi16(__W,__M,__A,__B); 
}
__m512i test_mm512_maskz_packus_epi16(__mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_packus_epi16
  // CHECK: @llvm.x86.avx512.mask.packuswb.512
  return _mm512_maskz_packus_epi16(__M,__A,__B); 
}
__m512i test_mm512_adds_epi8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_adds_epi8
  // CHECK: @llvm.x86.avx512.mask.padds.b.512
  return _mm512_adds_epi8(__A,__B); 
}
__m512i test_mm512_mask_adds_epi8(__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_adds_epi8
  // CHECK: @llvm.x86.avx512.mask.padds.b.512
  return _mm512_mask_adds_epi8(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_adds_epi8(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_adds_epi8
  // CHECK: @llvm.x86.avx512.mask.padds.b.512
  return _mm512_maskz_adds_epi8(__U,__A,__B); 
}
__m512i test_mm512_adds_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_adds_epi16
  // CHECK: @llvm.x86.avx512.mask.padds.w.512
  return _mm512_adds_epi16(__A,__B); 
}
__m512i test_mm512_mask_adds_epi16(__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_adds_epi16
  // CHECK: @llvm.x86.avx512.mask.padds.w.512
  return _mm512_mask_adds_epi16(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_adds_epi16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_adds_epi16
  // CHECK: @llvm.x86.avx512.mask.padds.w.512
  return _mm512_maskz_adds_epi16(__U,__A,__B); 
}
__m512i test_mm512_adds_epu8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_adds_epu8
  // CHECK: @llvm.x86.avx512.mask.paddus.b.512
  return _mm512_adds_epu8(__A,__B); 
}
__m512i test_mm512_mask_adds_epu8(__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_adds_epu8
  // CHECK: @llvm.x86.avx512.mask.paddus.b.512
  return _mm512_mask_adds_epu8(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_adds_epu8(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_adds_epu8
  // CHECK: @llvm.x86.avx512.mask.paddus.b.512
  return _mm512_maskz_adds_epu8(__U,__A,__B); 
}
__m512i test_mm512_adds_epu16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_adds_epu16
  // CHECK: @llvm.x86.avx512.mask.paddus.w.512
  return _mm512_adds_epu16(__A,__B); 
}
__m512i test_mm512_mask_adds_epu16(__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_adds_epu16
  // CHECK: @llvm.x86.avx512.mask.paddus.w.512
  return _mm512_mask_adds_epu16(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_adds_epu16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_adds_epu16
  // CHECK: @llvm.x86.avx512.mask.paddus.w.512
  return _mm512_maskz_adds_epu16(__U,__A,__B); 
}
__m512i test_mm512_avg_epu8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_avg_epu8
  // CHECK: @llvm.x86.avx512.mask.pavg.b.512
  return _mm512_avg_epu8(__A,__B); 
}
__m512i test_mm512_mask_avg_epu8(__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_avg_epu8
  // CHECK: @llvm.x86.avx512.mask.pavg.b.512
  return _mm512_mask_avg_epu8(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_avg_epu8(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_avg_epu8
  // CHECK: @llvm.x86.avx512.mask.pavg.b.512
  return _mm512_maskz_avg_epu8(__U,__A,__B); 
}
__m512i test_mm512_avg_epu16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_avg_epu16
  // CHECK: @llvm.x86.avx512.mask.pavg.w.512
  return _mm512_avg_epu16(__A,__B); 
}
__m512i test_mm512_mask_avg_epu16(__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_avg_epu16
  // CHECK: @llvm.x86.avx512.mask.pavg.w.512
  return _mm512_mask_avg_epu16(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_avg_epu16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_avg_epu16
  // CHECK: @llvm.x86.avx512.mask.pavg.w.512
  return _mm512_maskz_avg_epu16(__U,__A,__B); 
}
__m512i test_mm512_max_epi8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_max_epi8
  // CHECK: @llvm.x86.avx512.mask.pmaxs.b.512
  return _mm512_max_epi8(__A,__B); 
}
__m512i test_mm512_maskz_max_epi8(__mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_max_epi8
  // CHECK: @llvm.x86.avx512.mask.pmaxs.b.512
  return _mm512_maskz_max_epi8(__M,__A,__B); 
}
__m512i test_mm512_mask_max_epi8(__m512i __W, __mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_max_epi8
  // CHECK: @llvm.x86.avx512.mask.pmaxs.b.512
  return _mm512_mask_max_epi8(__W,__M,__A,__B); 
}
__m512i test_mm512_max_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_max_epi16
  // CHECK: @llvm.x86.avx512.mask.pmaxs.w.512
  return _mm512_max_epi16(__A,__B); 
}
__m512i test_mm512_maskz_max_epi16(__mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_max_epi16
  // CHECK: @llvm.x86.avx512.mask.pmaxs.w.512
  return _mm512_maskz_max_epi16(__M,__A,__B); 
}
__m512i test_mm512_mask_max_epi16(__m512i __W, __mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_max_epi16
  // CHECK: @llvm.x86.avx512.mask.pmaxs.w.512
  return _mm512_mask_max_epi16(__W,__M,__A,__B); 
}
__m512i test_mm512_max_epu8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_max_epu8
  // CHECK: @llvm.x86.avx512.mask.pmaxu.b.512
  return _mm512_max_epu8(__A,__B); 
}
__m512i test_mm512_maskz_max_epu8(__mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_max_epu8
  // CHECK: @llvm.x86.avx512.mask.pmaxu.b.512
  return _mm512_maskz_max_epu8(__M,__A,__B); 
}
__m512i test_mm512_mask_max_epu8(__m512i __W, __mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_max_epu8
  // CHECK: @llvm.x86.avx512.mask.pmaxu.b.512
  return _mm512_mask_max_epu8(__W,__M,__A,__B); 
}
__m512i test_mm512_max_epu16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_max_epu16
  // CHECK: @llvm.x86.avx512.mask.pmaxu.w.512
  return _mm512_max_epu16(__A,__B); 
}
__m512i test_mm512_maskz_max_epu16(__mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_max_epu16
  // CHECK: @llvm.x86.avx512.mask.pmaxu.w.512
  return _mm512_maskz_max_epu16(__M,__A,__B); 
}
__m512i test_mm512_mask_max_epu16(__m512i __W, __mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_max_epu16
  // CHECK: @llvm.x86.avx512.mask.pmaxu.w.512
  return _mm512_mask_max_epu16(__W,__M,__A,__B); 
}
__m512i test_mm512_min_epi8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_min_epi8
  // CHECK: @llvm.x86.avx512.mask.pmins.b.512
  return _mm512_min_epi8(__A,__B); 
}
__m512i test_mm512_maskz_min_epi8(__mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_min_epi8
  // CHECK: @llvm.x86.avx512.mask.pmins.b.512
  return _mm512_maskz_min_epi8(__M,__A,__B); 
}
__m512i test_mm512_mask_min_epi8(__m512i __W, __mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_min_epi8
  // CHECK: @llvm.x86.avx512.mask.pmins.b.512
  return _mm512_mask_min_epi8(__W,__M,__A,__B); 
}
__m512i test_mm512_min_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_min_epi16
  // CHECK: @llvm.x86.avx512.mask.pmins.w.512
  return _mm512_min_epi16(__A,__B); 
}
__m512i test_mm512_maskz_min_epi16(__mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_min_epi16
  // CHECK: @llvm.x86.avx512.mask.pmins.w.512
  return _mm512_maskz_min_epi16(__M,__A,__B); 
}
__m512i test_mm512_mask_min_epi16(__m512i __W, __mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_min_epi16
  // CHECK: @llvm.x86.avx512.mask.pmins.w.512
  return _mm512_mask_min_epi16(__W,__M,__A,__B); 
}
__m512i test_mm512_min_epu8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_min_epu8
  // CHECK: @llvm.x86.avx512.mask.pminu.b.512
  return _mm512_min_epu8(__A,__B); 
}
__m512i test_mm512_maskz_min_epu8(__mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_min_epu8
  // CHECK: @llvm.x86.avx512.mask.pminu.b.512
  return _mm512_maskz_min_epu8(__M,__A,__B); 
}
__m512i test_mm512_mask_min_epu8(__m512i __W, __mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_min_epu8
  // CHECK: @llvm.x86.avx512.mask.pminu.b.512
  return _mm512_mask_min_epu8(__W,__M,__A,__B); 
}
__m512i test_mm512_min_epu16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_min_epu16
  // CHECK: @llvm.x86.avx512.mask.pminu.w.512
  return _mm512_min_epu16(__A,__B); 
}
__m512i test_mm512_maskz_min_epu16(__mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_min_epu16
  // CHECK: @llvm.x86.avx512.mask.pminu.w.512
  return _mm512_maskz_min_epu16(__M,__A,__B); 
}
__m512i test_mm512_mask_min_epu16(__m512i __W, __mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_min_epu16
  // CHECK: @llvm.x86.avx512.mask.pminu.w.512
  return _mm512_mask_min_epu16(__W,__M,__A,__B); 
}
__m512i test_mm512_shuffle_epi8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shuffle_epi8
  // CHECK: @llvm.x86.avx512.mask.pshuf.b.512
  return _mm512_shuffle_epi8(__A,__B); 
}
__m512i test_mm512_mask_shuffle_epi8(__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shuffle_epi8
  // CHECK: @llvm.x86.avx512.mask.pshuf.b.512
  return _mm512_mask_shuffle_epi8(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_shuffle_epi8(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shuffle_epi8
  // CHECK: @llvm.x86.avx512.mask.pshuf.b.512
  return _mm512_maskz_shuffle_epi8(__U,__A,__B); 
}
__m512i test_mm512_subs_epi8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_subs_epi8
  // CHECK: @llvm.x86.avx512.mask.psubs.b.512
  return _mm512_subs_epi8(__A,__B); 
}
__m512i test_mm512_mask_subs_epi8(__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_subs_epi8
  // CHECK: @llvm.x86.avx512.mask.psubs.b.512
  return _mm512_mask_subs_epi8(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_subs_epi8(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_subs_epi8
  // CHECK: @llvm.x86.avx512.mask.psubs.b.512
  return _mm512_maskz_subs_epi8(__U,__A,__B); 
}
__m512i test_mm512_subs_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_subs_epi16
  // CHECK: @llvm.x86.avx512.mask.psubs.w.512
  return _mm512_subs_epi16(__A,__B); 
}
__m512i test_mm512_mask_subs_epi16(__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_subs_epi16
  // CHECK: @llvm.x86.avx512.mask.psubs.w.512
  return _mm512_mask_subs_epi16(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_subs_epi16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_subs_epi16
  // CHECK: @llvm.x86.avx512.mask.psubs.w.512
  return _mm512_maskz_subs_epi16(__U,__A,__B); 
}
__m512i test_mm512_subs_epu8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_subs_epu8
  // CHECK: @llvm.x86.avx512.mask.psubus.b.512
  return _mm512_subs_epu8(__A,__B); 
}
__m512i test_mm512_mask_subs_epu8(__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_subs_epu8
  // CHECK: @llvm.x86.avx512.mask.psubus.b.512
  return _mm512_mask_subs_epu8(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_subs_epu8(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_subs_epu8
  // CHECK: @llvm.x86.avx512.mask.psubus.b.512
  return _mm512_maskz_subs_epu8(__U,__A,__B); 
}
__m512i test_mm512_subs_epu16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_subs_epu16
  // CHECK: @llvm.x86.avx512.mask.psubus.w.512
  return _mm512_subs_epu16(__A,__B); 
}
__m512i test_mm512_mask_subs_epu16(__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_subs_epu16
  // CHECK: @llvm.x86.avx512.mask.psubus.w.512
  return _mm512_mask_subs_epu16(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_subs_epu16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_subs_epu16
  // CHECK: @llvm.x86.avx512.mask.psubus.w.512
  return _mm512_maskz_subs_epu16(__U,__A,__B); 
}
__m512i test_mm512_mask2_permutex2var_epi16(__m512i __A, __m512i __I, __mmask32 __U, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask2_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.mask.vpermi2var.hi.512
  return _mm512_mask2_permutex2var_epi16(__A,__I,__U,__B); 
}
__m512i test_mm512_permutex2var_epi16(__m512i __A, __m512i __I, __m512i __B) {
  // CHECK-LABEL: @test_mm512_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.hi.512
  return _mm512_permutex2var_epi16(__A,__I,__B); 
}
__m512i test_mm512_mask_permutex2var_epi16(__m512i __A, __mmask32 __U, __m512i __I, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.hi.512
  return _mm512_mask_permutex2var_epi16(__A,__U,__I,__B); 
}
__m512i test_mm512_maskz_permutex2var_epi16(__mmask32 __U, __m512i __A, __m512i __I, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.hi.512
  return _mm512_maskz_permutex2var_epi16(__U,__A,__I,__B); 
}
