// REQUIRES: disabled
// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -fasm-blocks -Wno-microsoft -verify -fsyntax-only

void t1(void) { 
 __asm __asm // expected-error {{__asm used with no assembly instructions}}
}

void f() {
  int foo;
  __asm { 
    mov eax, eax
    .unknowndirective // expected-error {{unknown directive}}
  }
  f();
  __asm {
    mov eax, 1+=2 // expected-error 2 {{unknown token in expression}}
  }
  f();
  __asm {
    mov eax, 1+++ // expected-error 2 {{unknown token in expression}}
  }
  f();
  __asm {
    mov eax, LENGTH bar // expected-error {{Unable to lookup expr!}}
  }
  f();
  __asm {
    mov eax, SIZE bar // expected-error {{Unable to lookup expr!}}
  }
  f();
  __asm {
    mov eax, TYPE bar // expected-error {{Unable to lookup expr!}}
  }
}
