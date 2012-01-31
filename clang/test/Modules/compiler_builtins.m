// RUN: rm -rf %t
// RUN: %clang -fsyntax-only -fmodules -fmodule-cache-path %t %s -Xclang -verify

@import _Builtin_stdlib.float_constants;

float getFltMax() { return FLT_MAX; }

@import _Builtin_stdlib.limits;

char getCharMax() { return CHAR_MAX; }

size_t size; // expected-error{{unknown type name 'size_t'}}

@import _Builtin_stdlib.stdint;

intmax_t value;

#ifdef __SSE__
@import _Builtin_intrinsics.intel.sse;
#endif

#ifdef __AVX2__
@import _Builtin_intrinsics.intel.avx2;
#endif
