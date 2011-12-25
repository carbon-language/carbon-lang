// RUN: %clang_cc1 %s -O3 -triple=x86_64-apple-darwin -target-feature +bmi -S -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

unsigned short test__tzcnt16(unsigned short __X)
{
  // CHECK: tzcntw
  return __tzcnt16(__X);
}

unsigned int test_tzcnt32(unsigned int __X)
{
  // CHECK: tzcntl
  return __tzcnt32(__X);
}

unsigned long long test__tzcnt64(unsigned long long __X)
{
  // CHECK: tzcntq
  return __tzcnt64(__X);
}
