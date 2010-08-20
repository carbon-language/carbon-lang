// RUN: %clang -fsyntax-only %s
// RUN: %clang -fsyntax-only -fno-lax-vector-conversions %s
// RUN: %clangxx -fsyntax-only -x c++ %s

#if defined(i386) || defined(__x86_64__)

#ifdef __MMX__
#include <mm_malloc.h>
#endif

#ifdef __SSE4_2__
// nmmintrin forwards to smmintrin.
#include <nmmintrin.h>
#endif

// immintrin includes all other intel intrinsic headers.
#include <immintrin.h>

#endif
