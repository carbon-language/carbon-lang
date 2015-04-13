// RUN: %clang_cc1 -triple i686-pc-linux-gnu -emit-llvm-only %s -verify
// RUN: %clang_cc1 -triple i686-pc-linux-gnu -femit-all-decls -emit-llvm-only %s -verify

void foo(void *p) __asm("_ZN1SC2Ev");
void foo(void *p) { } // expected-note {{previous}}

struct S {
  S() {} // expected-error {{definition with same mangled name as another definition}}
} s;
