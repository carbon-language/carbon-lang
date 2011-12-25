// RUN: %clang_cc1 %s -O3 -triple=x86_64-apple-darwin -target-feature +lzcnt -S -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

unsigned short test__lzcnt16(unsigned short __X)
{
  // CHECK: lzcntw
  return __lzcnt16(__X);
}

unsigned int test_lzcnt32(unsigned int __X)
{
  // CHECK: lzcntl
  return __lzcnt32(__X);
}

unsigned long long test__lzcnt64(unsigned long long __X)
{
  // CHECK: lzcntq
  return __lzcnt64(__X);
}
