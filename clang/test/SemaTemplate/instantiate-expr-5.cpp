// RUN: %clang_cc1 -fsyntax-only -verify %s

template <class A> int x(A x) { return x++; }
int y() { return x<int>(1); }

namespace PR5880 {
  template<typename T>
  struct A { 
    static const int a  = __builtin_offsetof(T, a.array[5].m); // expected-error{{error: no member named 'a' in 'HasM'}}
  };
  struct HasM {
    float m;
  };

  struct ArrayOfHasM {
    HasM array[10];
  };

  struct B { ArrayOfHasM a; };
  A<B> x;
  A<HasM> x2; // expected-note{{in instantiation of}}
}
