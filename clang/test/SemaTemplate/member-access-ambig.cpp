// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR8439
class A
{
};

class B
{
public:
  A & m;
};

class Base
{
public:
  B &f();
};

class Derived1 : public Base { };

class Derived2 : public Base { };

class X : public B, public Derived2, public Derived1
{
public:
  virtual void g();
};

void X::g()
{
  m.f<int>(); // expected-error{{no member named 'f' in 'A'}} \
  // expected-error{{expected '(' for function-style cast}} \
  // expected-error{{expected expression}}
}

namespace PR11134 {
  template<typename Derived> class A;
  template<typename Derived> class B : A<Derived> {
    typedef A<Derived> Base;
    using Base::member;
    int member;
  };
}

