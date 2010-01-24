// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s
namespace Constructor {
struct A {
  A(int);
};

struct B {
  explicit B(int);
};

B::B(int) { }

struct C {
  void f(const A&);
  void f(const B&);
};

void f(C c) {
  c.f(10);
}
}

namespace Conversion {
  struct A {
    operator int();
    explicit operator bool();
  };

  A::operator bool() { return false; } 

  struct B {
    void f(int);
    void f(bool);
  };

  void f(A a, B b) {
    b.f(a);
  }
}
