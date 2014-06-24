// RUN: %clang_cc1 -triple i686-win32 -verify -std=c++11 %s

extern int __attribute__((dllimport)) y;
constexpr int *x = &y; // expected-error {{must be initialized by a constant expression}}
