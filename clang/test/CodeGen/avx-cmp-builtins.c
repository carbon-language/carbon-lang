// RUN: %clang -mavx -c -emit-llvm %s -o - | llvm-dis | FileCheck %s
#include <immintrin.h>

//
// Test if third argument of cmp_XY function in LLVM IR form has immediate value.
//
void test_cmp_ps256() {
    __m256 a, b, c;
    a = _mm256_cmp_ps(b, c, _CMP_GE_OS);
    // CHECK: @test_cmp_ps256
    // CHECK: %0 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %tmp, <8 x float> %tmp1, i8 13)
}

void test_cmp_pd256() {
    __m256d a, b, c;
    a = _mm256_cmp_pd(b, c, _CMP_GE_OS);
    // CHECK: @test_cmp_pd256
    // CHECK: %0 = call <4 x double> @llvm.x86.avx.cmp.pd.256(<4 x double> %tmp, <4 x double> %tmp1, i8 13)
}

void test_cmp_ps() {
    __m128 a, b, c;
    a = _mm_cmp_ps(b, c, _CMP_GE_OS);
    // CHECK: @test_cmp_ps
    // CHECK: %cmpps = call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> %tmp, <4 x float> %tmp1, i8 13)
}

void test_cmp_pd() {
    __m128d a, b, c;
    a = _mm_cmp_pd(b, c, _CMP_GE_OS);
    // CHECK: @test_cmp_pd
    // CHECK: %cmppd = call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %tmp, <2 x double> %tmp1, i8 13)
}

void test_cmp_sd() {
    __m128d a, b, c;
    a = _mm_cmp_sd(b, c, _CMP_GE_OS);
    // CHECK: @test_cmp_sd
    // CHECK: %cmpsd = call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %tmp, <2 x double> %tmp1, i8 13)
}

void test_cmp_ss() {
    __m128 a, b, c;
    a = _mm_cmp_ss(b, c, _CMP_GE_OS);
    // CHECK: @test_cmp_ss
    // CHECK: %cmpss = call <4 x float> @llvm.x86.sse.cmp.ss(<4 x float> %tmp, <4 x float> %tmp1, i8 13)
}
