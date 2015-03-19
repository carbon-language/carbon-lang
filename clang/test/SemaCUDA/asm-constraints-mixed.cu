// RUN: %clang_cc1 -triple nvptx-unknown-cuda -fsyntax-only -fcuda-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// expected-no-diagnostics

__attribute__((device)) void df() {
  short h;
  // asm with PTX constraints. Some of them are PTX-specific.
  __asm__("dont care" : "=h"(h): "f"(0.0), "d"(0.0), "h"(0), "r"(0), "l"(0));
}

void hf() {
  int a;
  // Asm with x86 constraints that are not supported by PTX.
  __asm__("dont care" : "=a"(a): "a"(0), "b"(0), "c"(0));
}
