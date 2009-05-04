// RUN: clang-cc -fsyntax-only -verify %s

class Base { };
class Derived1 : public Base { };
class Derived2 : public Base { };

void f0(volatile Base *b, Derived1 *d1, const Derived2 *d2) {
  if (b > d1)
    return;
  if (d1 <= b)
    return;
  if (b > d2)
    return;
  if (d1 >= d2) // expected-error{{comparison of distinct}}
    return;
}

void f1(volatile Base *b, Derived1 *d1, const Derived2 *d2) {
  if (b == d1)
    return;
  if (d1 == b)
    return;
  if (b != d2)
    return;
  if (d1 == d2) // expected-error{{comparison of distinct}}
    return;
}
