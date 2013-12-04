// REQUIRES: x86-registered-target
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
    mov eax, 1+=2 // expected-error {{unknown token in expression}}
  }
  f();
  __asm {
    mov eax, 1+++ // expected-error {{unknown token in expression}}
  }
  f();
  __asm {
    mov eax, LENGTH bar // expected-error {{unable to lookup expression}}
  }
  f();
  __asm {
    mov eax, SIZE bar // expected-error {{unable to lookup expression}}
  }
  f();
  __asm {
    mov eax, TYPE bar // expected-error {{unable to lookup expression}}
  }
}

void rdar15318432(void) {
  // We used to crash on this.  When LLVM called back to Clang to parse a name
  // and do name lookup, if parsing failed, we did not restore the lexer state
  // properly.

  // expected-error@+2 {{expected identifier}}
  __asm {
    and ecx, ~15
  }

  int x = 0;
  // expected-error@+3 {{expected identifier}}
  __asm {
    and ecx, x
    and ecx, ~15
  }
}
