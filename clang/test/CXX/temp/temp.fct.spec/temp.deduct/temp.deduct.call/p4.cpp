// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace PR8598 {
  template<class T> struct identity { typedef T type; };

  template<class T, class C>
  void f(T C::*, typename identity<T>::type*){}
  
  struct X { void f() {}; };
  
  void g() { (f)(&X::f, 0); }
}
