// RUN: %clangxx_msan -m64 -O0 %s -c -o %t
// RUN: %clangxx_msan -m64 -O3 %s -c -o %t

// Regression test for MemorySanitizer instrumentation of a select instruction
// with vector arguments.

#include <emmintrin.h>

__m128d select(bool b, __m128d c, __m128d d)
{
  return b ? c : d;
}

