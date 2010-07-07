// RUN: %clang_cc1 -fsyntax-only -Wunused-variable -verify %s
template<typename T> void f() {
  T t;
  t = 17;
}

// PR5407
struct A { A(); };
struct B { ~B(); };
void f() {
  A a;
  B b;
}

// PR5531
namespace PR5531 {
  struct A {
  };

  struct B {
    B(int);
  };

  struct C {
    ~C();
  };

  void test() {
    A(); // expected-warning{{expression result unused}}
    B(17);
    C();
  }
}


struct X {
 int foo() __attribute__((warn_unused_result));
};

void bah() {
  X x, *x2;
  x.foo(); // expected-warning {{ignoring return value of function declared with warn_unused_result attribute}}
  x2->foo(); // expected-warning {{ignoring return value of function declared with warn_unused_result attribute}}
}

template<typename T>
struct X0 { };

template<typename T>
void test_dependent_init(T *p) {
  X0<int> i(p);
  (void)i;
}

namespace PR6948 {
  template<typename T> class X;
  
  void f() {
    X<char> str (read_from_file()); // expected-error{{use of undeclared identifier 'read_from_file'}}
  }
}
