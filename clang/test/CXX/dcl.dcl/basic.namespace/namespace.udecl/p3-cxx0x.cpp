// RUN: clang-cc -fsyntax-only -verify %s
// C++0x N2914.

struct B {
  void f(char);
  void g(char);
  enum E { e };
  union { int x; };
};

class C {
  int g();
};

class D2 : public B {
  using B::f;
  using B::e;
  using B::x;
  using C::g; // expected-error{{using declaration refers into 'C::', which is not a base class of 'D2'}}
};
