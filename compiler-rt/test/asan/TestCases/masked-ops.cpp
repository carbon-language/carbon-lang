// We need optimization to convert target-specific masked loads/stores to llvm
// generic loads/stores
// RUN: %clangxx_asan -o %t %s -mavx -O1
// RUN: not %run %t l1 2>&1 | FileCheck -check-prefix=CHECK-L1 %s
// RUN: %run %t l6 | FileCheck -check-prefix=CHECK-L6 %s
// RUN: %run %t la | FileCheck -check-prefix=CHECK-LA %s
// RUN: not %run %t s1 2>&1 | FileCheck -check-prefix=CHECK-S1 %s
// RUN: %run %t s6 | FileCheck -check-prefix=CHECK-S6 %s
// RUN: %run %t sa | FileCheck -check-prefix=CHECK-SA %s
// REQUIRES: x86-target-arch
#include <assert.h>
#include <stdio.h>
#include <x86intrin.h>

float g_vec3[3] = {1802398064.0, 1881171305.0, 25961.0};

void maskedstore_ps_0110(__m128 a) {
  _mm_maskstore_ps(g_vec3, (__v4si){0, -1, -1, 0}, a);
}

void maskedstore_ps_0001(__m128 a) {
  _mm_maskstore_ps(g_vec3, (__v4si){0, 0, 0, -1}, a);
}

void maskedstore_ps_1010(__m128 a) {
  _mm_maskstore_ps(g_vec3, (__v4si){-1, 0, -1, 0}, a);
}

__m128i maskedload_ps_0110() {
  return _mm_maskload_ps(g_vec3, (__v4si){0, -1, -1, 0});
}

__m128i maskedload_ps_0001() {
  return _mm_maskload_ps(g_vec3, (__v4si){0, 0, 0, -1});
}

__m128i maskedload_ps_1010() {
  return _mm_maskload_ps(g_vec3, (__v4si){-1, 0, -1, 0});
}

__m128 a = (__v4sf){1.0, 2.0, 3.0, 4.0};

void print_vector(__v4sf v) {
  printf("%d,%d,%d,%d\n", (int)v[0], (int)v[1], (int)v[2], (int)v[3]);
}
void print_vector3(float * v) {
  printf("%d,%d,%d\n", (int)v[0], (int)v[1], (int)v[2]);
}

int main(int argc, char **argv) {
  assert(argc > 1);
  bool isLoad = argv[1][0] == 'l';
  assert(isLoad || argv[1][0] == 's');
  if (isLoad) {
    switch (argv[1][1]) {
    case '1': {
      // CHECK-L1: ERROR: AddressSanitizer
      __v4sf v = maskedload_ps_0001();
      print_vector(v);
      // Should have blown up
      break;
    }
    case '6': {
      // Safe
      __v4sf v = maskedload_ps_0110();
      // CHECK-L6: 0,1881171328,25961,0
      print_vector(v);
      return 0;
    }
    case 'a': {
      // TODO: Poison middle element
      // Safe
      __v4sf v = maskedload_ps_1010();
      // CHECK-LA: 1802398080,0,25961,0
      print_vector(v);
      return 0;
    }
    }
  } else {
    switch (argv[1][1]) {
    case '1':
      // CHECK-S1: ERROR: AddressSanitizer
      maskedstore_ps_0001(a);
      // Should have blown up
      break;
    case '6':
      // Safe
      maskedstore_ps_0110(a);
      // CHECK-S6: 1802398080,2,3
      print_vector3(g_vec3);
      return 0;
    case 'a':
      // TODO: Poison middle element
      // Safe
      maskedstore_ps_1010(a);
      // CHECK-SA: 1,1881171328,3
      print_vector3(g_vec3);
      return 0;
    }
  }
  assert(0);
}
