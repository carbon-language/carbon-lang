// RUN: %clang_cc1 -ffreestanding -triple x86_64-apple-macosx10.8.0 -target-feature +sse4.1 -g -emit-llvm %s -o - | FileCheck %s

#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

__m128 test_rsqrt_ss(__m128 x) {
  // CHECK: define {{.*}} @test_rsqrt_ss
  // CHECK: call <4 x float> @llvm.x86.sse.rsqrt.ss
  // CHECK: extractelement <4 x float> {{.*}}, i32 0
  // CHECK: extractelement <4 x float> {{.*}}, i32 1
  // CHECK: extractelement <4 x float> {{.*}}, i32 2
  // CHECK: extractelement <4 x float> {{.*}}, i32 3
  return _mm_rsqrt_ss(x);
}

__m128 test_rcp_ss(__m128 x) {
  // CHECK: define {{.*}} @test_rcp_ss
  // CHECK: call <4 x float> @llvm.x86.sse.rcp.ss
  // CHECK: extractelement <4 x float> {{.*}}, i32 0
  // CHECK: extractelement <4 x float> {{.*}}, i32 1
  // CHECK: extractelement <4 x float> {{.*}}, i32 2
  // CHECK: extractelement <4 x float> {{.*}}, i32 3
  return _mm_rcp_ss(x);
}

__m128 test_sqrt_ss(__m128 x) {
  // CHECK: define {{.*}} @test_sqrt_ss
  // CHECK: call <4 x float> @llvm.x86.sse.sqrt.ss
  // CHECK: extractelement <4 x float> {{.*}}, i32 0
  // CHECK: extractelement <4 x float> {{.*}}, i32 1
  // CHECK: extractelement <4 x float> {{.*}}, i32 2
  // CHECK: extractelement <4 x float> {{.*}}, i32 3
  return _mm_sqrt_ss(x);
}

__m128 test_loadl_pi(__m128 x, void* y) {
  // CHECK: define {{.*}} @test_loadl_pi
  // CHECK: load <2 x float>* {{.*}}, align 1{{$}}
  // CHECK: shufflevector {{.*}} <4 x i32> <i32 0, i32 1
  // CHECK: shufflevector {{.*}} <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  return _mm_loadl_pi(x,y);
}

__m128 test_loadh_pi(__m128 x, void* y) {
  // CHECK: define {{.*}} @test_loadh_pi
  // CHECK: load <2 x float>* {{.*}}, align 1{{$}}
  // CHECK: shufflevector {{.*}} <4 x i32> <i32 0, i32 1
  // CHECK: shufflevector {{.*}} <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  return _mm_loadh_pi(x,y);
}

__m128 test_load_ss(void* y) {
  // CHECK: define {{.*}} @test_load_ss
  // CHECK: load float* {{.*}}, align 1{{$}}
  return _mm_load_ss(y);
}

__m128 test_load1_ps(void* y) {
  // CHECK: define {{.*}} @test_load1_ps
  // CHECK: load float* {{.*}}, align 1{{$}}
  return _mm_load1_ps(y);
}

void test_store_ss(__m128 x, void* y) {
  // CHECK-LABEL: define void @test_store_ss
  // CHECK: store {{.*}} float* {{.*}}, align 1,
  _mm_store_ss(y, x);
}

__m128d test_load1_pd(__m128 x, void* y) {
  // CHECK: define {{.*}} @test_load1_pd
  // CHECK: load double* {{.*}}, align 1{{$}}
  return _mm_load1_pd(y);
}

__m128d test_loadr_pd(__m128 x, void* y) {
  // CHECK: define {{.*}} @test_loadr_pd
  // CHECK: load <2 x double>* {{.*}}, align 16{{$}}
  return _mm_loadr_pd(y);
}

__m128d test_load_sd(void* y) {
  // CHECK: define {{.*}} @test_load_sd
  // CHECK: load double* {{.*}}, align 1{{$}}
  return _mm_load_sd(y);
}

__m128d test_loadh_pd(__m128d x, void* y) {
  // CHECK: define {{.*}} @test_loadh_pd
  // CHECK: load double* {{.*}}, align 1{{$}}
  return _mm_loadh_pd(x, y);
}

__m128d test_loadl_pd(__m128d x, void* y) {
  // CHECK: define {{.*}} @test_loadl_pd
  // CHECK: load double* {{.*}}, align 1{{$}}
  return _mm_loadl_pd(x, y);
}

void test_store_sd(__m128d x, void* y) {
  // CHECK-LABEL: define void @test_store_sd
  // CHECK: store {{.*}} double* {{.*}}, align 1{{$}}
  _mm_store_sd(y, x);
}

void test_store1_pd(__m128d x, void* y) {
  // CHECK-LABEL: define void @test_store1_pd
  // CHECK: store {{.*}} double* {{.*}}, align 1{{$}}
  // CHECK: store {{.*}} double* {{.*}}, align 1{{$}}
  _mm_store1_pd(y, x);
}

void test_storer_pd(__m128d x, void* y) {
  // CHECK-LABEL: define void @test_storer_pd
  // CHECK: store {{.*}} <2 x double>* {{.*}}, align 16{{$}}
  _mm_storer_pd(y, x);
}

void test_storeh_pd(__m128d x, void* y) {
  // CHECK-LABEL: define void @test_storeh_pd
  // CHECK: store {{.*}} double* {{.*}}, align 1{{$}}
  _mm_storeh_pd(y, x);
}

void test_storel_pd(__m128d x, void* y) {
  // CHECK-LABEL: define void @test_storel_pd
  // CHECK: store {{.*}} double* {{.*}}, align 1{{$}}
  _mm_storel_pd(y, x);
}

__m128i test_loadl_epi64(void* y) {
  // CHECK: define {{.*}} @test_loadl_epi64
  // CHECK: load i64* {{.*}}, align 1{{$}}
  return _mm_loadl_epi64(y);
}

__m128i test_mm_minpos_epu16(__m128i x) {
  // CHECK: define {{.*}} @test_mm_minpos_epu16
  // CHECK: @llvm.x86.sse41.phminposuw
  return _mm_minpos_epu16(x);
}

__m128i test_mm_mpsadbw_epu8(__m128i x, __m128i y) {
  // CHECK: define {{.*}} @test_mm_mpsadbw_epu8
  // CHECK: @llvm.x86.sse41.mpsadbw
  return _mm_mpsadbw_epu8(x, y, 1);
}

__m128 test_mm_dp_ps(__m128 x, __m128 y) {
  // CHECK: define {{.*}} @test_mm_dp_ps
  // CHECK: @llvm.x86.sse41.dpps
  return _mm_dp_ps(x, y, 2);
}

__m128d test_mm_dp_pd(__m128d x, __m128d y) {
  // CHECK: define {{.*}} @test_mm_dp_pd
  // CHECK: @llvm.x86.sse41.dppd
  return _mm_dp_pd(x, y, 2);
}

__m128 test_mm_round_ps(__m128 x) {
  // CHECK: define {{.*}} @test_mm_round_ps
  // CHECK: @llvm.x86.sse41.round.ps
  return _mm_round_ps(x, 2);
}

__m128 test_mm_round_ss(__m128 x, __m128 y) {
  // CHECK: define {{.*}} @test_mm_round_ss
  // CHECK: @llvm.x86.sse41.round.ss
  return _mm_round_ss(x, y, 2);
}

__m128d test_mm_round_pd(__m128d x) {
  // CHECK: define {{.*}} @test_mm_round_pd
  // CHECK: @llvm.x86.sse41.round.pd
  return _mm_round_pd(x, 2);
}

__m128d test_mm_round_sd(__m128d x, __m128d y) {
  // CHECK: define {{.*}} @test_mm_round_sd
  // CHECK: @llvm.x86.sse41.round.sd
  return _mm_round_sd(x, y, 2);
}

void test_storel_epi64(__m128i x, void* y) {
  // CHECK-LABEL: define void @test_storel_epi64
  // CHECK: store {{.*}} i64* {{.*}}, align 1{{$}}
  _mm_storel_epi64(y, x);
}

void test_stream_si32(int x, void *y) {
  // CHECK-LABEL: define void @test_stream_si32
  // CHECK: store {{.*}} i32* {{.*}}, align 1, !nontemporal
  _mm_stream_si32(y, x);
}

void test_stream_si64(long long x, void *y) {
  // CHECK-LABEL: define void @test_stream_si64
  // CHECK: store {{.*}} i64* {{.*}}, align 1, !nontemporal
  _mm_stream_si64(y, x);
}

void test_stream_si128(__m128i x, void *y) {
  // CHECK-LABEL: define void @test_stream_si128
  // CHECK: store {{.*}} <2 x i64>* {{.*}}, align 16, !nontemporal
  _mm_stream_si128(y, x);
}

void test_extract_epi16(__m128i __a) {
  // CHECK-LABEL: define void @test_extract_epi16
  // CHECK: [[x:%.*]] = and i32 %{{.*}}, 7
  // CHECK: extractelement <8 x i16> %{{.*}}, i32 [[x]]
  _mm_extract_epi16(__a, 8);
}

int test_extract_ps(__m128i __a) {
  // CHECK-LABEL: @test_extract_ps
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  return _mm_extract_ps(__a, 4);
}

int test_extract_epi8(__m128i __a) {
  // CHECK-LABEL: @test_extract_epi8
  // CHECK: extractelement <16 x i8> %{{.*}}, i32 0
  return _mm_extract_epi8(__a, 16);
}

int test_extract_epi32(__m128i __a) {
  // CHECK-LABEL: @test_extract_epi32
  // CHECK: extractelement <4 x i32> %{{.*}}, i32 0
  return _mm_extract_epi32(__a, 4);
}

void test_insert_epi32(__m128i __a, int b) {
  // CHECK-LABEL: @test_insert_epi32
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 0
   _mm_insert_epi32(__a, b, 4);
}

__m128d test_blend_pd(__m128d V1, __m128d V2) {
  // CHECK-LABEL: @test_blend_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 2, i32 1>
  return _mm_blend_pd(V1, V2, 1);
}

__m128 test_blend_ps(__m128 V1, __m128 V2) {
  // CHECK-LABEL: @test_blend_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  return _mm_blend_ps(V1, V2, 5);
}

__m128i test_blend_epi16(__m128i V1, __m128i V2) {
  // CHECK-LABEL: @test_blend_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 7>
  return _mm_blend_epi16(V1, V2, 42);
}
