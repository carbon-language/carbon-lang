// RUN: %clang_cc1 -triple i686-win32 -verify -std=c++11 %s

extern int __attribute__((dllimport)) var;
constexpr int *varp = &var; // expected-error {{must be initialized by a constant expression}}

extern __attribute__((dllimport)) void fun();
constexpr void (*funp)(void) = &fun; // expected-error {{must be initialized by a constant expression}}
