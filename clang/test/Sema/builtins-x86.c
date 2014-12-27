// RUN: %clang_cc1 -triple=x86_64-apple-darwin -fsyntax-only -verify %s

typedef float __m128 __attribute__((__vector_size__(16)));
typedef double __m128d __attribute__((__vector_size__(16)));

__m128 test__builtin_ia32_cmpps(__m128 __a, __m128 __b) {
  __builtin_ia32_cmpps(__a, __b, 32); // expected-error {{argument should be a value from 0 to 31}}
}

__m128d test__builtin_ia32_cmppd(__m128d __a, __m128d __b) {
  __builtin_ia32_cmppd(__a, __b, 32); // expected-error {{argument should be a value from 0 to 31}}
}

__m128 test__builtin_ia32_cmpss(__m128 __a, __m128 __b) {
  __builtin_ia32_cmpss(__a, __b, 32); // expected-error {{argument should be a value from 0 to 31}}
}

__m128d test__builtin_ia32_cmpsd(__m128d __a, __m128d __b) {
  __builtin_ia32_cmpsd(__a, __b, 32); // expected-error {{argument should be a value from 0 to 31}}
}
