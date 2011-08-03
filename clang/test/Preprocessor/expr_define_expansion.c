// RUN: %clang_cc1 %s -E -CC -pedantic -verify

#define FOO && 1
#if defined FOO FOO
#endif
