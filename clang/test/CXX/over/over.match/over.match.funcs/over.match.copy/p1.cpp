// RUN: %clang_cc1 -std=c++11 -fsyntax-only %s -verify

namespace ExplicitConv {
  struct X { }; // expected-note 2{{candidate constructor}}

  struct Y {
    explicit operator X() const;
  };

  void test(const Y& y) {
    X x(static_cast<X>(y));
    X x2((X)y);
    X x3 = y; // expected-error{{no viable conversion from 'const ExplicitConv::Y' to 'ExplicitConv::X'}}
  }
}
