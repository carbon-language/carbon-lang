// RUN: %clang_cc1 %s -triple i686-apple-darwin -verify -fsyntax-only -fno-gnu-inline-asm

void f (void) {
  long long foo = 0, bar;
  asm volatile("INST %0, %1" : "=r"(foo) : "r"(bar)); // expected-error {{GNU-style inline assembly is disabled}}
  return;
}
