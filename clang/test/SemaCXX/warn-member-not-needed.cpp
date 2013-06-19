// RUN: %clang_cc1 -fsyntax-only -verify -Wunneeded-member-function %s

namespace {
  class A {
    void g() {} // expected-warning {{is not needed and will not be emitted}}
    template <typename T>
    void foo() {
      g();
    }
  };
}
