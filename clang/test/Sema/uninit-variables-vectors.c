// RUN: %clang_cc1 -fsyntax-only -Wuninitialized -fsyntax-only %s -verify

#include <xmmintrin.h>

void test1(float *input) {
  __m128 x, y, z, w, X; // expected-note {{variable 'x' is declared here}} expected-note {{variable 'y' is declared here}} expected-note {{variable 'w' is declared here}}  expected-note {{variable 'z' is declared here}}
  x = _mm_xor_ps(x,x); // expected-warning {{variable 'x' is possibly uninitialized when used here}}
  y = _mm_xor_ps(y,y); // expected-warning {{variable 'y' is possibly uninitialized when used here}}
  z = _mm_xor_ps(z,z); // expected-warning {{variable 'z' is possibly uninitialized when used here}}
  w = _mm_xor_ps(w,w); // expected-warning {{variable 'w' is possibly uninitialized when used here}}
  X = _mm_loadu_ps(&input[0]);
  X = _mm_xor_ps(X,X); // no-warning
}

