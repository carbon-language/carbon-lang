// RUN: %clang_cc1 %s -O3 -triple=x86_64-apple-darwin -target-feature +avx2 -emit-llvm -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

__m256 test_mm256_mpsadbw_epu8(__m256 x, __m256 y) {
  // CHECK: @llvm.x86.avx2.mpsadbw({{.*}}, {{.*}}, i32 3)
  return _mm256_mpsadbw_epu8(x, y, 3);
}

__m256 test_mm256_abs_epi8(__m256 a) {
  // CHECK: @llvm.x86.avx2.pabs.b
  return _mm256_abs_epi8(a);
}

__m256 test_mm256_abs_epi16(__m256 a) {
  // CHECK: @llvm.x86.avx2.pabs.w
  return _mm256_abs_epi16(a);
}

__m256 test_mm256_abs_epi32(__m256 a) {
  // CHECK: @llvm.x86.avx2.pabs.d
  return _mm256_abs_epi32(a);
}

__m256 test_mm256_packs_epi16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.packsswb
  return _mm256_packs_epi16(a, b);
}

__m256 test_mm256_packs_epi32(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.packssdw
  return _mm256_packs_epi32(a, b);
}

__m256 test_mm256_packs_epu16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.packuswb
  return _mm256_packus_epi16(a, b);
}

__m256 test_mm256_packs_epu32(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.packusdw
  return _mm256_packus_epi32(a, b);
}

__m256 test_mm256_add_epi8(__m256 a, __m256 b) {
  // CHECK: add <32 x i8>
  return _mm256_add_epi8(a, b);
}

__m256 test_mm256_add_epi16(__m256 a, __m256 b) {
  // CHECK: add <16 x i16>
  return _mm256_add_epi16(a, b);
}

__m256 test_mm256_add_epi32(__m256 a, __m256 b) {
  // CHECK: add <8 x i32>
  return _mm256_add_epi32(a, b);
}

__m256 test_mm256_add_epi64(__m256 a, __m256 b) {
  // CHECK: add <4 x i64>
  return _mm256_add_epi64(a, b);
}

__m256 test_mm256_adds_epi8(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.padds.b
  return _mm256_adds_epi8(a, b);
}

__m256 test_mm256_adds_epi16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.padds.w
  return _mm256_adds_epi16(a, b);
}

__m256 test_mm256_adds_epu8(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.paddus.b
  return _mm256_adds_epu8(a, b);
}

__m256 test_mm256_adds_epu16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.paddus.w
  return _mm256_adds_epu16(a, b);
}

__m256 test_mm256_alignr_epi8(__m256 a, __m256 b) {
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49>
  return _mm256_alignr_epi8(a, b, 2);
}

__m256 test2_mm256_alignr_epi8(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.psrl.dq({{.*}}, i32 8)
  return _mm256_alignr_epi8(a, b, 17);
}

__m256 test_mm256_sub_epi8(__m256 a, __m256 b) {
  // CHECK: sub <32 x i8>
  return _mm256_sub_epi8(a, b);
}

__m256 test_mm256_sub_epi16(__m256 a, __m256 b) {
  // CHECK: sub <16 x i16>
  return _mm256_sub_epi16(a, b);
}

__m256 test_mm256_sub_epi32(__m256 a, __m256 b) {
  // CHECK: sub <8 x i32>
  return _mm256_sub_epi32(a, b);
}

__m256 test_mm256_sub_epi64(__m256 a, __m256 b) {
  // CHECK: sub <4 x i64>
  return _mm256_sub_epi64(a, b);
}

__m256 test_mm256_subs_epi8(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.psubs.b
  return _mm256_subs_epi8(a, b);
}

__m256 test_mm256_subs_epi16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.psubs.w
  return _mm256_subs_epi16(a, b);
}

__m256 test_mm256_subs_epu8(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.psubus.b
  return _mm256_subs_epu8(a, b);
}

__m256 test_mm256_subs_epu16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.psubus.w
  return _mm256_subs_epu16(a, b);
}

__m256 test_mm256_and_si256(__m256 a, __m256 b) {
  // CHECK: and <4 x i64>
  return _mm256_and_si256(a, b);
}

__m256 test_mm256_andnot_si256(__m256 a, __m256 b) {
  // CHECK: xor <4 x i64>
  // CHECK: and <4 x i64>
  return _mm256_andnot_si256(a, b);
}

__m256 test_mm256_or_si256(__m256 a, __m256 b) {
  // CHECK: or <4 x i64>
  return _mm256_or_si256(a, b);
}

__m256 test_mm256_xor_si256(__m256 a, __m256 b) {
  // CHECK: xor <4 x i64>
  return _mm256_xor_si256(a, b);
}

__m256 test_mm256_avg_epu8(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pavg.b
  return _mm256_avg_epu8(a, b);
}

__m256 test_mm256_avg_epu16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pavg.w
  return _mm256_avg_epu16(a, b);
}

__m256 test_mm256_blendv_epi8(__m256 a, __m256 b, __m256 m) {
  // CHECK: @llvm.x86.avx2.pblendvb
  return _mm256_blendv_epi8(a, b, m);
}

__m256 test_mm256_blend_epi16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pblendw(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, i32 2)
  return _mm256_blend_epi16(a, b, 2);
}

__m256 test_mm256_cmpeq_epi8(__m256 a, __m256 b) {
  // CHECK: icmp eq <32 x i8>
  return _mm256_cmpeq_epi8(a, b);
}

__m256 test_mm256_cmpeq_epi16(__m256 a, __m256 b) {
  // CHECK: icmp eq <16 x i16>
  return _mm256_cmpeq_epi16(a, b);
}

__m256 test_mm256_cmpeq_epi32(__m256 a, __m256 b) {
  // CHECK: icmp eq <8 x i32>
  return _mm256_cmpeq_epi32(a, b);
}

__m256 test_mm256_cmpeq_epi64(__m256 a, __m256 b) {
  // CHECK: icmp eq <4 x i64>
  return _mm256_cmpeq_epi64(a, b);
}

__m256 test_mm256_cmpgt_epi8(__m256 a, __m256 b) {
  // CHECK: icmp sgt <32 x i8>
  return _mm256_cmpgt_epi8(a, b);
}

__m256 test_mm256_cmpgt_epi16(__m256 a, __m256 b) {
  // CHECK: icmp sgt <16 x i16>
  return _mm256_cmpgt_epi16(a, b);
}

__m256 test_mm256_cmpgt_epi32(__m256 a, __m256 b) {
  // CHECK: icmp sgt <8 x i32>
  return _mm256_cmpgt_epi32(a, b);
}

__m256 test_mm256_cmpgt_epi64(__m256 a, __m256 b) {
  // CHECK: icmp sgt <4 x i64>
  return _mm256_cmpgt_epi64(a, b);
}

__m256 test_mm256_hadd_epi16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.phadd.w
  return _mm256_hadd_epi16(a, b);
}

__m256 test_mm256_hadd_epi32(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.phadd.d
  return _mm256_hadd_epi32(a, b);
}

__m256 test_mm256_hadds_epi16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.phadd.sw
  return _mm256_hadds_epi16(a, b);
}

__m256 test_mm256_hsub_epi16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.phsub.w
  return _mm256_hsub_epi16(a, b);
}

__m256 test_mm256_hsub_epi32(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.phsub.d
  return _mm256_hsub_epi32(a, b);
}

__m256 test_mm256_hsubs_epi16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.phsub.sw
  return _mm256_hsubs_epi16(a, b);
}

__m256 test_mm256_maddubs_epi16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pmadd.ub.sw
  return _mm256_maddubs_epi16(a, b);
}

__m256 test_mm256_madd_epi16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pmadd.wd
  return _mm256_madd_epi16(a, b);
}

__m256 test_mm256_max_epi8(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pmaxs.b
  return _mm256_max_epi8(a, b);
}

__m256 test_mm256_max_epi16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pmaxs.w
  return _mm256_max_epi16(a, b);
}

__m256 test_mm256_max_epi32(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pmaxs.d
  return _mm256_max_epi32(a, b);
}

__m256 test_mm256_max_epu8(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pmaxu.b
  return _mm256_max_epu8(a, b);
}

__m256 test_mm256_max_epu16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pmaxu.w
  return _mm256_max_epu16(a, b);
}

__m256 test_mm256_max_epu32(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pmaxu.d
  return _mm256_max_epu32(a, b);
}

__m256 test_mm256_min_epi8(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pmins.b
  return _mm256_min_epi8(a, b);
}

__m256 test_mm256_min_epi16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pmins.w
  return _mm256_min_epi16(a, b);
}

__m256 test_mm256_min_epi32(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pmins.d
  return _mm256_min_epi32(a, b);
}

__m256 test_mm256_min_epu8(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pminu.b
  return _mm256_min_epu8(a, b);
}

__m256 test_mm256_min_epu16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pminu.w
  return _mm256_min_epu16(a, b);
}

__m256 test_mm256_min_epu32(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pminu.d
  return _mm256_min_epu32(a, b);
}

int test_mm256_movemask_epi8(__m256 a) {
  // CHECK: @llvm.x86.avx2.pmovmskb
  return _mm256_movemask_epi8(a);
}

__m256 test_mm256_cvtepi8_epi16(__m128 a) {
  // CHECK: @llvm.x86.avx2.pmovsxbw
  return _mm256_cvtepi8_epi16(a);
}

__m256 test_mm256_cvtepi8_epi32(__m128 a) {
  // CHECK: @llvm.x86.avx2.pmovsxbd
  return _mm256_cvtepi8_epi32(a);
}

__m256 test_mm256_cvtepi8_epi64(__m128 a) {
  // CHECK: @llvm.x86.avx2.pmovsxbq
  return _mm256_cvtepi8_epi64(a);
}

__m256 test_mm256_cvtepi16_epi32(__m128 a) {
  // CHECK: @llvm.x86.avx2.pmovsxwd
  return _mm256_cvtepi16_epi32(a);
}

__m256 test_mm256_cvtepi16_epi64(__m128 a) {
  // CHECK: @llvm.x86.avx2.pmovsxwq
  return _mm256_cvtepi16_epi64(a);
}

__m256 test_mm256_cvtepi32_epi64(__m128 a) {
  // CHECK: @llvm.x86.avx2.pmovsxdq
  return _mm256_cvtepi32_epi64(a);
}

__m256 test_mm256_cvtepu8_epi16(__m128 a) {
  // CHECK: @llvm.x86.avx2.pmovzxbw
  return _mm256_cvtepu8_epi16(a);
}

__m256 test_mm256_cvtepu8_epi32(__m128 a) {
  // CHECK: @llvm.x86.avx2.pmovzxbd
  return _mm256_cvtepu8_epi32(a);
}

__m256 test_mm256_cvtepu8_epi64(__m128 a) {
  // CHECK: @llvm.x86.avx2.pmovzxbq
  return _mm256_cvtepu8_epi64(a);
}

__m256 test_mm256_cvtepu16_epi32(__m128 a) {
  // CHECK: @llvm.x86.avx2.pmovzxwd
  return _mm256_cvtepu16_epi32(a);
}

__m256 test_mm256_cvtepu16_epi64(__m128 a) {
  // CHECK: @llvm.x86.avx2.pmovzxwq
  return _mm256_cvtepu16_epi64(a);
}

__m256 test_mm256_cvtepu32_epi64(__m128 a) {
  // CHECK: @llvm.x86.avx2.pmovzxdq
  return _mm256_cvtepu32_epi64(a);
}

__m256 test_mm256_mul_epi32(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pmul.dq
  return _mm256_mul_epi32(a, b);
}

__m256 test_mm256_mulhrs_epi16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pmul.hr.sw
  return _mm256_mulhrs_epi16(a, b);
}

__m256 test_mm256_mulhi_epu16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pmulhu.w
  return _mm256_mulhi_epu16(a, b);
}

__m256 test_mm256_mulhi_epi16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pmulh.w
  return _mm256_mulhi_epi16(a, b);
}

__m256 test_mm256_mullo_epi16(__m256 a, __m256 b) {
  // CHECK: mul <16 x i16>
  return _mm256_mullo_epi16(a, b);
}

__m256 test_mm256_mullo_epi32(__m256 a, __m256 b) {
  // CHECK: mul <8 x i32>
  return _mm256_mullo_epi32(a, b);
}

__m256 test_mm256_mul_epu32(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.pmulu.dq
  return _mm256_mul_epu32(a, b);
}
