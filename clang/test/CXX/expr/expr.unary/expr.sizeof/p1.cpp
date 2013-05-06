// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A {
  unsigned bitX : 4;
  unsigned bitY : 4;
  unsigned var;

  void foo();
};

void test(A *a) {
  int x;
  x = sizeof(a->bitX); // expected-error {{invalid application of 'sizeof' to bit-field}}
  x = sizeof((unsigned) a->bitX);
  x = sizeof(a->foo(), a->bitX); // expected-error {{invalid application of 'sizeof' to bit-field}}
  x = sizeof(a->var ? a->bitX : a->bitY); // expected-error {{invalid application of 'sizeof' to bit-field}}
  x = sizeof(a->var ? a->bitX : a->bitX); // expected-error {{invalid application of 'sizeof' to bit-field}}
  x = sizeof(a->bitX = 3); // expected-error {{invalid application of 'sizeof' to bit-field}}
  x = sizeof(a->bitY += 3); // expected-error {{invalid application of 'sizeof' to bit-field}}
}
