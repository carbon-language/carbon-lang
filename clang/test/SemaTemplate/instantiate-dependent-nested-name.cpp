// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR4382
template<typename T> struct X { static const T A = 1; };
template<typename T, bool = X<T>::A> struct Y { typedef T A; };
template<typename T> struct Z { typedef typename Y<T>::A A; };
extern int x;
extern Z<int>::A x;

namespace pr21964 {
struct H;
template <class> struct T {
  struct A; // expected-note {{member is declared here}}
  static void B() {
    A::template N<H>; // expected-error {{implicit instantiation of undefined member 'pr21964::T<pr21964::H>::A'}}
  }
};
template struct T<H>; // expected-note {{requested here}}
}
