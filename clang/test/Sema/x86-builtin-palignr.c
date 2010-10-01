// RUN: %clang_cc1 -fsyntax-only -target-feature +ssse3 -verify %s
// Temporarily xfail this on windows.
// XFAIL: win32

#include <tmmintrin.h>

__m64 test1(__m64 a, __m64 b, int c) {
   return _mm_alignr_pi8(a, b, c); // expected-error {{argument to '__builtin_ia32_palignr' must be a constant integer}}
}

int test2(int N) {
 __m128i white2;
 white2 = __builtin_ia32_pslldqi128(white2, N); // expected-error {{argument to '__builtin_ia32_pslldqi128' must be a constant integer}}
 return 0;
} 

