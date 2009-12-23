// RUN: %clang_cc1 -fsyntax-only -verify %s

friend class A; // expected-error {{'friend' used outside of class}}
void f() { friend class A; } // expected-error {{'friend' used outside of class}}
class C { friend class A; };
class D { void f() { friend class A; } }; // expected-error {{'friend' used outside of class}}

// PR5760
namespace test0 {
  namespace ns {
    void f(int);
  }

  struct A {
    friend void ns::f(int a);
  };
}

// Test derived from LLVM's Registry.h
namespace test1 {
  template <class T> struct Outer {
    void foo(T);
    struct Inner {
      friend void Outer::foo(T);
    };
  };

  void test() {
    (void) Outer<int>::Inner();
  }
}

// PR5476
namespace test2 {
  namespace foo {
    void Func(int x);
  }

  class Bar {
    friend void ::test2::foo::Func(int x);
  };
}
