// RUN: %clang -fsyntax-only %s
// RUN: %clang -fsyntax-only -fno-lax-vector-conversions %s
// RUN: %clang -fsyntax-only -x c++ %s

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

#endif
