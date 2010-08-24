// RUN: %clang_cc1 -fsyntax-only -verify %s

// rdar://problem/8347416
namespace test0 {
  struct A {
    void foo(void (A::*)(int)); // expected-note {{passing argument to parameter here}}
    template<typename T> void g(T);

    void test() {
      // FIXME: this diagnostic is terrible
      foo(&g<int>); // expected-error {{cannot initialize a parameter of type 'void (test0::A::*)(int)' with an rvalue of type '<overloaded function type>'}}
    }
  };
}

// This should succeed.
namespace test1 {
  struct A {
    static void f(void (A::*)());
    static void f(void (*)(int));
    void g();
    static void g(int);

    void test() {
      f(&g);
    }
  };
}

// Also rdar://problem/8347416
namespace test2 {
  struct A {
    static int foo(short);
    static int foo(float);
    int foo(int);
    int foo(double);

    void test();
  };

  void A::test() {
    // FIXME: This diagnostic is terrible.
    int (A::*ptr)(int) = &(A::foo); // expected-error {{cannot initialize a variable of type 'int (test2::A::*)(int)' with an rvalue of type '<overloaded function type>'}}
  }
}
