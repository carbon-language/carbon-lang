// RUN: %clang_cc1 -triple i386-apple-darwin9 -target-cpu pentium4 -target-feature +sse4.1 -g -emit-llvm %s -o - | FileCheck %s

#include <emmintrin.h>

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
  // CHECK: define void @test_store_ss
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
  // CHECK: define void @test_store_sd
  // CHECK: store {{.*}} double* {{.*}}, align 1{{$}}
  _mm_store_sd(y, x);
}

void test_store1_pd(__m128d x, void* y) {
  // CHECK: define void @test_store1_pd
  // CHECK: store {{.*}} double* {{.*}}, align 1{{$}}
  // CHECK: store {{.*}} double* {{.*}}, align 1{{$}}
  _mm_store1_pd(y, x);
}

void test_storer_pd(__m128d x, void* y) {
  // CHECK: define void @test_storer_pd
  // CHECK: store {{.*}} <2 x double>* {{.*}}, align 16{{$}}
  _mm_storer_pd(y, x);
}

void test_storeh_pd(__m128d x, void* y) {
  // CHECK: define void @test_storeh_pd
  // CHECK: store {{.*}} double* {{.*}}, align 1{{$}}
  _mm_storeh_pd(y, x);
}

void test_storel_pd(__m128d x, void* y) {
  // CHECK: define void @test_storel_pd
  // CHECK: store {{.*}} double* {{.*}}, align 1{{$}}
  _mm_storel_pd(y, x);
}

__m128i test_loadl_epi64(void* y) {
  // CHECK: define {{.*}} @test_loadl_epi64
  // CHECK: load i64* {{.*}}, align 1{{$}}
  return _mm_loadl_epi64(y);
}
