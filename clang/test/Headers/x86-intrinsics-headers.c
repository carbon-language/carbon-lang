// RUN: %clang -fsyntax-only %s
// RUN: %clang -fsyntax-only -fno-lax-vector-conversions %s
// RUN: %clangxx -fsyntax-only -x c++ %s

#if defined(i386) || defined(__x86_64__)

#  if defined(__MMX__)
#include <emmintrin.h>
#include <mm_malloc.h>
#  endif

#  if defined(__SSE__)
#include <xmmintrin.h>
#  endif

#  if defined(__SSE3__)
#include <pmmintrin.h>
#  endif

#  if defined(__SSSE3__)
#include <tmmintrin.h>
#  endif

#  if defined(__SSE4_1__)
#include <smmintrin.h>
#  endif

#  if defined(__SSE4_2__)
#include <nmmintrin.h>
#  endif

#  if defined(__AVX__)
#include <avxintrin.h>
#  endif

#endif
