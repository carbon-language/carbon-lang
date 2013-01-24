// RUN: %clang_cc1 %s -triple powerpc64-unknown-linux-gnu -fasm-blocks -verify -fsyntax-only

void f() {
  __asm nop // expected-error {{Unsupported architecture 'powerpc64' for MS-style inline assembly}}
}
