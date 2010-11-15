// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

// Core DR 532.
namespace PR8130 {
  struct A { };

  template<class T> struct B {
    template<class R> int &operator*(R&);
  };

  template<class T, class R> float &operator*(T&, R&);
  void test() {
    A a;
    B<A> b;
    int &ir = b * a;
  }
}
