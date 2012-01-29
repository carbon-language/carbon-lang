// RUN: rm -rf %t
// RUN: %clang -fsyntax-only -fmodules -fmodule-cache-path %t %s -Xclang -verify

// Note: disable test on MSVC, which doesn't have <complex.h>.
// REQUIRES: shell

@import __compiler_builtins.float_constants;

float getFltMax() { return FLT_MAX; }

@import __compiler_builtins.limits;

char getCharMax() { return CHAR_MAX; }

size_t size; // expected-error{{unknown type name 'size_t'}}
