// RUN: %clang_cc1 -fsyntax-only -target-feature +ssse3 -verify %s
// Temporarily xfail this on windows.
// XFAIL: win32

#include <tmmintrin.h>

__m64 foo(__m64 a, __m64 b, int c)
{
   return _mm_alignr_pi8(a, b, c); // expected-error {{argument to '__builtin_ia32_palignr' must be a constant integer}}
}
