// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-feature +avx -emit-llvm %s -o - | FileCheck %s
#include <immintrin.h>

//
// Test if third argument of cmp_XY function in LLVM IR form has immediate value.
//
void test_cmp_ps256() {
    __m256 a, b, c;
    a = _mm256_cmp_ps(b, c, _CMP_GE_OS);
    // CHECK: @test_cmp_ps256
    // CHECK: call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> {{%.*}}, <8 x float> {{%.*}}, i8 13)
}

void test_cmp_pd256() {
    __m256d a, b, c;
    a = _mm256_cmp_pd(b, c, _CMP_GE_OS);
    // CHECK: @test_cmp_pd256
    // CHECK: call <4 x double> @llvm.x86.avx.cmp.pd.256(<4 x double> {{%.*}}, <4 x double> {{%.*}}, i8 13)
}

void test_cmp_ps() {
    __m128 a, b, c;
    a = _mm_cmp_ps(b, c, _CMP_GE_OS);
    // CHECK: @test_cmp_ps
    // CHECK: call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> {{%.*}}, <4 x float> {{%.*}}, i8 13)
}

void test_cmp_pd() {
    __m128d a, b, c;
    a = _mm_cmp_pd(b, c, _CMP_GE_OS);
    // CHECK: @test_cmp_pd
    // CHECK: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> {{%.*}}, <2 x double> {{%.*}}, i8 13)
}

void test_cmp_sd() {
    __m128d a, b, c;
    a = _mm_cmp_sd(b, c, _CMP_GE_OS);
    // CHECK: @test_cmp_sd
    // CHECK: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> {{%.*}}, <2 x double> {{%.*}}, i8 13)
}

void test_cmp_ss() {
    __m128 a, b, c;
    a = _mm_cmp_ss(b, c, _CMP_GE_OS);
    // CHECK: @test_cmp_ss
    // CHECK: call <4 x float> @llvm.x86.sse.cmp.ss(<4 x float> {{%.*}}, <4 x float> {{%.*}}, i8 13)
}
