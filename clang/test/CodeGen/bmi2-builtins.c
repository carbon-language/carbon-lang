// RUN: %clang_cc1 %s -O3 -triple=x86_64-apple-darwin -target-feature +bmi2 -emit-llvm -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

unsigned int test_bzhi_u32(unsigned int __X, unsigned int __Y) {
  // CHECK: @llvm.x86.bmi.bzhi.32
  return _bzhi_u32(__X, __Y);
}

unsigned int test_pdep_u32(unsigned int __X, unsigned int __Y) {
  // CHECK: @llvm.x86.bmi.pdep.32
  return _pdep_u32(__X, __Y);
}

unsigned int test_pext_u32(unsigned int __X, unsigned int __Y) {
  // CHECK: @llvm.x86.bmi.pext.32
  return _pext_u32(__X, __Y);
}

unsigned long long test_bzhi_u64(unsigned long long __X, unsigned long long __Y) {
  // CHECK: @llvm.x86.bmi.bzhi.64
  return _bzhi_u64(__X, __Y);
}

unsigned long long test_pdep_u64(unsigned long long __X, unsigned long long __Y) {
  // CHECK: @llvm.x86.bmi.pdep.64
  return _pdep_u64(__X, __Y);
}

unsigned long long test_pext_u64(unsigned long long __X, unsigned long long __Y) {
  // CHECK: @llvm.x86.bmi.pext.64
  return _pext_u64(__X, __Y);
}
