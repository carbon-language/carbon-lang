// RUN: %clang_cc1 -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -fsyntax-only %s
int a;

inline __attribute__((noreturn(a))) void *f1();  // expected-error {{'noreturn' attribute takes no arguments}}
inline __attribute__((always_inline(a))) void *f2();  // expected-error {{'always_inline' attribute takes no arguments}}
inline __attribute__((cdecl(a))) void *f3();  // expected-error {{'cdecl' attribute takes no arguments}}
inline __attribute__((const(a))) void *f4();  // expected-error {{'const' attribute takes no arguments}}
inline __attribute__((fastcall(a))) void *f5();  // expected-error {{'fastcall' attribute takes no arguments}}
inline __attribute__((malloc(a))) void *f5();  // expected-error {{'malloc' attribute takes no arguments}}
inline __attribute__((nothrow(a))) void *f7();  // expected-error {{'nothrow' attribute takes no arguments}}
inline __attribute__((stdcall(a))) void *f8();  // expected-error {{'stdcall' attribute takes no arguments}}
inline __attribute__((used(a))) void *f9();  // expected-error {{'used' attribute takes no arguments}}
inline __attribute__((unused(a))) void *f10();  // expected-error {{'unused' attribute takes no arguments}}
inline __attribute__((weak(a))) void *f11();  // expected-error {{'weak' attribute takes no arguments}}
