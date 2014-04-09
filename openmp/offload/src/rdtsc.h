//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

uint64_t _rdtsc()
{
  uint32_t eax, edx;
  asm volatile ("rdtsc" : "=a" (eax), "=d" (edx));
  return ((uint64_t)edx << 32) | eax;
}
