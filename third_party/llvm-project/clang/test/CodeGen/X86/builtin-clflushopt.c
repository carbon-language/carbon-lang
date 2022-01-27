// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-apple-darwin -target-feature +clflushopt  -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

void test_mm_clflushopt(char * __m) {
  //CHECK-LABEL: @test_mm_clflushopt
  //CHECK: @llvm.x86.clflushopt
  _mm_clflushopt(__m);
}
