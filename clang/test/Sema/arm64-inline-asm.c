// RUN: %clang_cc1 -triple arm64-apple-ios7.1 -fsyntax-only -verify %s
// expected-no-diagnostics

void foo() {
  asm volatile("USE(%0)" :: "z"(0LL));
  asm volatile("USE(%x0)" :: "z"(0LL));
  asm volatile("USE(%w0)" :: "z"(0));

}
