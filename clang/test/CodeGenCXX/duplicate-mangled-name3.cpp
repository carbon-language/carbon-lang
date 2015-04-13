// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm-only %s -verify
// RUN: %clang_cc1 -triple %itanium_abi_triple -femit-all-decls -emit-llvm-only %s -verify

extern "C" {
  void _ZN1SC2Ev(void *p) { } // expected-note {{previous}}
}

struct S {
  S() {} // expected-error {{definition with same mangled name as another definition}}
} s;
