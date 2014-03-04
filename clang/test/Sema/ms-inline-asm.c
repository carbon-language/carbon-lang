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

static int global;

int t2(int *arr, int i) {
  __asm {
    mov eax, arr;
    mov eax, arr[0];
    mov eax, arr[1 + 2];
    mov eax, arr[1 + (2 * 5) - 3 + 1<<1];
  }

  // expected-error@+1 {{cannot use base register with variable reference}}
  __asm mov eax, arr[ebp + 1 + (2 * 5) - 3 + 1<<1]
  // expected-error@+1 {{cannot use index register with variable reference}}
  __asm mov eax, arr[esi * 4]
  // expected-error@+1 {{cannot use more than one symbol in memory operand}}
  __asm mov eax, arr[i]
  // expected-error@+1 {{cannot use more than one symbol in memory operand}}
  __asm mov eax, global[i]

  // FIXME: Why don't we diagnose this?
  // expected-Xerror@+1 {{cannot reference multiple local variables in assembly operand}}
  //__asm mov eax, [arr + i];
  return 0;
}
