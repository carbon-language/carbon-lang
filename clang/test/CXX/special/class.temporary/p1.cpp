// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

namespace test0 {
  struct A {
    A() = default;
    int x;
    int y;

    A(const A&) = delete; // expected-note {{function has been explicitly marked deleted here}}
  };

  void foo(...);

  void test() {
    A a;
    foo(a); // expected-error {{call to deleted constructor of 'test0::A'}}
  }
}

namespace test1 {
  struct A {
    A() = default;
    int x;
    int y;

  private:
    A(const A&) = default; // expected-note {{declared private here}}
  };

  void foo(...);

  void test() {
    A a;
    // FIXME: this error about variadics is bogus
    foo(a); // expected-error {{calling a private constructor of class 'test1::A'}} expected-error {{cannot pass object of non-trivial type 'test1::A' through variadic function}}
  }
}
