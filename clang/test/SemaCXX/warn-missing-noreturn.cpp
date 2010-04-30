// RUN: %clang_cc1 -fsyntax-only -verify %s -Wmissing-noreturn
void f() __attribute__((noreturn));

template<typename T> void g(T) { // expected-warning {{function could be attribute 'noreturn'}}
  f();
}

template void g<int>(int); // expected-note {{in instantiation of function template specialization 'g<int>' requested here}}

template<typename T> struct A {
  void g() { // expected-warning {{function could be attribute 'noreturn'}}
    f();
  }
};

template struct A<int>; // expected-note {{in instantiation of member function 'A<int>::g' requested here}}

struct B {
  template<typename T> void g(T) { // expected-warning {{function could be attribute 'noreturn'}}
    f();
  }
};

template void B::g<int>(int); // expected-note {{in instantiation of function template specialization 'B::g<int>' requested here}}

// We don't want a warning here.
struct X {
  virtual void g() { f(); }
};

namespace test1 {
  bool condition();

  // We don't want a warning here.
  void foo() {
    while (condition()) {}
  }
}
