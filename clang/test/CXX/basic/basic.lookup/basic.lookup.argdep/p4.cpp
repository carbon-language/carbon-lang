// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace A {
  class A {
    friend void func(A);
    friend A operator+(A,A);
  };
}

namespace B {
  class B {
    static void func(B);
  };
  B operator+(B,B);
}

namespace D {
  class D {};
}

namespace C {
  class C {};
  void func(C);
  C operator+(C,C);
  D::D operator+(D::D,D::D);
}

namespace D {
  using namespace C;
}

namespace Test {
  void test() {
    func(A::A());
    func(B::B()); // expected-error {{use of undeclared identifier 'func'}}
    func(C::C());
    A::A() + A::A();
    B::B() + B::B();
    C::C() + C::C();
    D::D() + D::D(); // expected-error {{ invalid operands to binary expression ('D::D' and 'D::D') }}
  }
}
