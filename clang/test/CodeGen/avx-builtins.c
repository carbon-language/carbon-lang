// RUN: %clang_cc1 %s -O3 -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

//
// Test LLVM IR codegen of shuffle instructions
//

__m256 test__mm256_loadu_ps(void* p) {
  // CHECK: load <8 x float>* %{{.*}}, align 1
  return _mm256_loadu_ps(p);
}

__m256d test__mm256_loadu_pd(void* p) {
  // CHECK: load <4 x double>* %{{.*}}, align 1
  return _mm256_loadu_pd(p);
}

__m256i test__mm256_loadu_si256(void* p) {
  // CHECK: load <4 x i64>* %{{.+}}, align 1
  return _mm256_loadu_si256(p);
}
