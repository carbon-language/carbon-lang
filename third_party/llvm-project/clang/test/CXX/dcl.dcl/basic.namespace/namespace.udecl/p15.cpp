// RUN: %clang_cc1 -std=c++11 -verify %s

struct B1 {
  B1(int); // expected-note {{candidate}}
};

struct B2 {
  B2(int); // expected-note {{candidate}}
};

struct D1 : B1, B2 {
  using B1::B1; // expected-note {{inherited here}}
  using B2::B2; // expected-note {{inherited here}}
};
D1 d1(0); // expected-error {{ambiguous}}

struct D2 : B1, B2 {
  using B1::B1;
  using B2::B2;
  D2(int);
};
D2 d2(0); // ok


// The emergent behavior of implicit special members is a bit odd when
// inheriting from multiple base classes.
namespace default_ctor {
  struct C;
  struct D;

  struct convert_to_D1 {
    operator D&&();
  };
  struct convert_to_D2 {
    operator D&&();
  };

  struct A {
    A();

    A(C &&);
    C &operator=(C&&);

    A(D &&);
    D &operator=(D&&); // expected-note {{candidate}}

    A(convert_to_D2); // expected-note {{candidate}}
  };

  struct B {
    B();

    B(C &&);
    C &operator=(C&&);

    B(D &&);
    D &operator=(D&&); // expected-note {{candidate}}

    B(convert_to_D2); // expected-note {{candidate}}
  };

  struct C : A, B {
    using A::A;
    using A::operator=;
    using B::B;
    using B::operator=;
  };
  struct D : A, B {
    using A::A; // expected-note {{inherited here}}
    using A::operator=;
    using B::B; // expected-note {{inherited here}}
    using B::operator=;

    D(int);
    D(const D&);
    D &operator=(const D&);
  };

  C c;
  void f(C c) {
    C c2(static_cast<C&&>(c));
    c = static_cast<C&&>(c);
  }

  // D does not declare D(), D(D&&), nor operator=(D&&), so the base class
  // versions are inherited.
  // Exception: we implicitly declare a default constructor so that we can
  // reason about its properties.
  D d; // ok, implicitly declared
  void f(D d) {
    D d2(static_cast<D&&>(d)); // ok, ignores inherited constructors
    D d3(convert_to_D1{}); // ok, ignores inherited constructors
    D d4(convert_to_D2{}); // expected-error {{ambiguous}}
    d = static_cast<D&&>(d); // expected-error {{ambiguous}}
  }

  struct Y;
  struct X {
    X();
    X(volatile Y &); // expected-note 3{{inherited constructor cannot be used to copy object}}
  } x;
  struct Y : X { using X::X; } volatile y;
  struct Z : Y { using Y::Y; } volatile z; // expected-note 4{{no known conversion}} expected-note 2{{would lose volatile}} expected-note 3{{requires 0}} expected-note 3{{inherited here}}
  Z z1(x); // expected-error {{no match}}
  Z z2(y); // expected-error {{no match}}
  Z z3(z); // expected-error {{no match}}
}
