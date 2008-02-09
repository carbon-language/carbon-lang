// RUN: clang -fsyntax-only -verify %s

void f1() {
  asm ("ret" : : :); // expected-error {{expected string literal}}
}

void f2() {
  asm("foo" : "=r" (a)); // expected-error {{use of undeclared identifier 'a'}}
  asm("foo" : : "r" (b)); // expected-error {{use of undeclared identifier 'b'}} 
}
