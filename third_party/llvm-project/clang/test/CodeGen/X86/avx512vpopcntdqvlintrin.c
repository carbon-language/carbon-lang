// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vpopcntdq -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m128i test_mm_popcnt_epi64(__m128i __A) {
  // CHECK-LABEL: @test_mm_popcnt_epi64
  // CHECK: @llvm.ctpop.v2i64
  return _mm_popcnt_epi64(__A);
}
__m128i test_mm_mask_popcnt_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_popcnt_epi64
  // CHECK: @llvm.ctpop.v2i64
  // CHECK: select <2 x i1> %{{.+}}, <2 x i64> %{{[0-9]+}}, <2 x i64> {{.*}}
  return _mm_mask_popcnt_epi64(__W, __U, __A);
}
__m128i test_mm_maskz_popcnt_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_popcnt_epi64
  // CHECK: @llvm.ctpop.v2i64
  // CHECK: select <2 x i1> %{{.+}}, <2 x i64> %{{[0-9]+}}, <2 x i64> {{.*}}
  return _mm_maskz_popcnt_epi64(__U, __A);
}
__m128i test_mm_popcnt_epi32(__m128i __A) {
  // CHECK-LABEL: @test_mm_popcnt_epi32
  // CHECK: @llvm.ctpop.v4i32
  return _mm_popcnt_epi32(__A);
}
__m128i test_mm_mask_popcnt_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_popcnt_epi32
  // CHECK: @llvm.ctpop.v4i32
  // CHECK: select <4 x i1> %{{.+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> {{.*}}
  return _mm_mask_popcnt_epi32(__W, __U, __A);
}
__m128i test_mm_maskz_popcnt_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_popcnt_epi32
  // CHECK: @llvm.ctpop.v4i32
  // CHECK: select <4 x i1> %{{.+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> {{.*}}
  return _mm_maskz_popcnt_epi32(__U, __A);
}

__m256i test_mm256_popcnt_epi64(__m256i __A) {
  // CHECK-LABEL: @test_mm256_popcnt_epi64
  // CHECK: @llvm.ctpop.v4i64
  return _mm256_popcnt_epi64(__A);
}
__m256i test_mm256_mask_popcnt_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_popcnt_epi64
  // CHECK: @llvm.ctpop.v4i64
  // CHECK: select <4 x i1> %{{.+}}, <4 x i64> %{{[0-9]+}}, <4 x i64> {{.*}}
  return _mm256_mask_popcnt_epi64(__W, __U, __A);
}
__m256i test_mm256_maskz_popcnt_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_popcnt_epi64
  // CHECK: @llvm.ctpop.v4i64
  // CHECK: select <4 x i1> %{{.+}}, <4 x i64> %{{[0-9]+}}, <4 x i64> {{.*}}
  return _mm256_maskz_popcnt_epi64(__U, __A);
}
__m256i test_mm256_popcnt_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm256_popcnt_epi32
  // CHECK: @llvm.ctpop.v8i32
  return _mm256_popcnt_epi32(__A);
}
__m256i test_mm256_mask_popcnt_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_popcnt_epi32
  // CHECK: @llvm.ctpop.v8i32
  // CHECK: select <8 x i1> %{{.+}}, <8 x i32> %{{[0-9]+}}, <8 x i32> {{.*}}
  return _mm256_mask_popcnt_epi32(__W, __U, __A);
}
__m256i test_mm256_maskz_popcnt_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_popcnt_epi32
  // CHECK: @llvm.ctpop.v8i32
  // CHECK: select <8 x i1> %{{.+}}, <8 x i32> %{{[0-9]+}}, <8 x i32> {{.*}}
  return _mm256_maskz_popcnt_epi32(__U, __A);
}
