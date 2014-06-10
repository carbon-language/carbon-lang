// RUN: %clang_cc1 -Wdate-time -Wno-builtin-macro-redefined %s -verify -E

__TIME__ // expected-warning {{expansion of date or time macro is not reproducible}}
__DATE__  // expected-warning {{expansion of date or time macro is not reproducible}}
__TIMESTAMP__ // expected-warning {{expansion of date or time macro is not reproducible}}

#define __TIME__
__TIME__
