// RUN: clang-cc -fsyntax-only -verify %s

class X {};

void test() {
  X x;

  x.int; // expected-error{{expected identifier}}
  x.~int(); // expected-error{{expected identifier}}
  x.operator; // expected-error{{missing type specifier after 'operator'}}
  x.operator typedef; // expected-error{{missing type specifier after 'operator'}}
}

void test2() {
  X *x;

  x->int; // expected-error{{expected identifier}}
  x->~int(); // expected-error{{expected identifier}}
  x->operator; // expected-error{{missing type specifier after 'operator'}}
  x->operator typedef; // expected-error{{missing type specifier after 'operator'}}
}
