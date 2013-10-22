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

namespace test3 {
  static void f();
  template<typename T>
  static void g() {
    f();
  }
  static void f() {
  }
  void h() {
    g<int>();
  }
}

namespace test4 {
  static void f();
  static void f();
  template<typename T>
  static void g() {
    f();
  }
  static void f() {
  }
  void h() {
    g<int>();
  }
}

namespace test4 {
  static void func();
  void bar() {
    void func();
    func();
  }
  static void func() {}
}
