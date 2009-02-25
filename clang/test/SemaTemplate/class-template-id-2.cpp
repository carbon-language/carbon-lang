// RUN: clang -fsyntax-only -verify %s
namespace N {
  template<typename T> class A; 

  template<> class A<int> { };

  class B : public A<int> { };
}

class C1 : public N::A<int> { };

class C2 : public N::A<float> { }; // expected-error{{base class has incomplete type}} \
           // FIXME: expected-note{{forward declaration of 'class A'}}

struct D1 {
  operator N::A<int>();
};

namespace N {
  struct D2 {
    operator A<int>();
  };
}
