// RUN: %clang_cc1 %s -O3 -triple=x86_64-apple-darwin -target-feature +bmi -S -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

unsigned short test__tzcnt16(unsigned short __X) {
  // CHECK: tzcntw
  return __tzcnt16(__X);
}

unsigned int test__andn_u32(unsigned int __X, unsigned int __Y) {
  // CHECK: andnl
  return __andn_u32(__X, __Y);
}

unsigned int test__bextr_u32(unsigned int __X, unsigned int __Y) {
  // CHECK: bextrl
  return __bextr_u32(__X, __Y);
}

unsigned int test__blsi_u32(unsigned int __X) {
  // CHECK: blsil
  return __blsi_u32(__X);
}

unsigned int test__blsmsk_u32(unsigned int __X) {
  // CHECK: blsmskl
  return __blsmsk_u32(__X);
}

unsigned int test__blsr_u32(unsigned int __X) {
  // CHECK: blsrl
  return __blsr_u32(__X);
}

unsigned int test_tzcnt32(unsigned int __X) {
  // CHECK: tzcntl
  return __tzcnt32(__X);
}

unsigned long long test__andn_u64(unsigned long __X, unsigned long __Y) {
  // CHECK: andnq
  return __andn_u64(__X, __Y);
}

unsigned long long test__bextr_u64(unsigned long __X, unsigned long __Y) {
  // CHECK: bextrq
  return __bextr_u64(__X, __Y);
}

unsigned long long test__blsi_u64(unsigned long long __X) {
  // CHECK: blsiq
  return __blsi_u64(__X);
}

unsigned long long test__blsmsk_u64(unsigned long long __X) {
  // CHECK: blsmskq
  return __blsmsk_u64(__X);
}

unsigned long long test__blsr_u64(unsigned long long __X) {
  // CHECK: blsrq
  return __blsr_u64(__X);
}

unsigned long long test__tzcnt64(unsigned long long __X) {
  // CHECK: tzcntq
  return __tzcnt64(__X);
}
