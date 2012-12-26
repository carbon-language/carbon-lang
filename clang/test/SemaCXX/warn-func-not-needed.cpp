// RUN: %clang_cc1 -fsyntax-only -verify -Wall %s

namespace test1 {
  static void f() {} // expected-warning {{is not needed and will not be emitted}}
  static void f();
  template <typename T>
  void foo() {
    f();
  }
}

namespace test2 {
  static void f() {}
  static void f();
  static void g() { f(); }
  void h() { g(); }
}
