// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR5741
namespace test0 {
  struct A {
    struct B { };
    struct C;
  };

  struct A::C : B { };
}

namespace test1 {
  struct Opaque1 {};
  struct Opaque2 {};

  struct A {
    struct B { B(Opaque1); };
  };
  struct B {
    B(Opaque2);
  };

  struct C : A, B {
    // Apparently the base-or-member lookup is actually ambiguous
    // without this qualification.
    C() : A(), test1::B(Opaque2()) {}
  };
}
