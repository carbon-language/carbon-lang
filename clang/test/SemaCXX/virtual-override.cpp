// RUN: clang-cc -fsyntax-only -verify %s

namespace T1 {

class A {
  virtual int f(); // expected-note{{overridden virtual function is here}}
};

class B : A {
  virtual void f(); // expected-error{{virtual function 'f' has a different return type ('void') than the function it overrides (which has return type 'int')}}
};

}
