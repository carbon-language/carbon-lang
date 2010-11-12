// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace PR8130 {
  struct A { };

  template<class T> struct B {
    template<class R> int &operator*(R&); // #1
  };

  template<class T, class R> float &operator*(T&, R&); // #2
  void test() {
    A a;
    B<A> b;
    int &ir = b * a; // calls #1a
  }
}
