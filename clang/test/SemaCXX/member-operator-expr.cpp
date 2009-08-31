// RUN: clang-cc -fsyntax-only -verify %s

class X {
public:
  int operator++();
  operator int();
};

void test() {
  X x;
  int i;

  i = x.operator++();
  i = x.operator int();
  x.operator--(); // expected-error{{no member named 'operator--'}}
  x.operator float(); // expected-error{{no member named 'operator float'}}
  x.operator; // expected-error{{missing type specifier after 'operator'}}
}

void test2() {
  X *x;
  int i;

  i = x->operator++();
  i = x->operator int();
  x->operator--(); // expected-error{{no member named 'operator--'}}
  x->operator float(); // expected-error{{no member named 'operator float'}}
  x->operator; // expected-error{{missing type specifier after 'operator'}}
}
