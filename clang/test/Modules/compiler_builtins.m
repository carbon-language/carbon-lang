// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodule-cache-path %t -verify %s

@import __compiler_builtins.float_constants;

float getFltMax() { return FLT_MAX; }

@import __compiler_builtins.limits;

char getCharMax() { return CHAR_MAX; }

size_t size; // expected-error{{unknown type name 'size_t'}}
