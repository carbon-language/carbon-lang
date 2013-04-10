// RUN: %clang_cc1 -std=c++11 -verify %s

template<int> struct X {};

// A[n inheriting] constructor [...] has the same access as the corresponding
// constructor [in the base class].
struct A {
public:
  A(X<0>) {}
protected:
  A(X<1>) {}
private:
  A(X<2>) {} // expected-note {{declared private here}}
  friend class FA;
};

struct B : A {
  using A::A; // expected-error {{private constructor}} expected-note {{implicitly declared protected here}}
  friend class FB;
};

B b0{X<0>{}};
B b1{X<1>{}}; // expected-error {{calling a protected constructor}}
B b2{X<2>{}}; // expected-note {{first required here}}

struct C : B {
  C(X<0> x) : B(x) {}
  C(X<1> x) : B(x) {}
};

struct FB {
  B b0{X<0>{}};
  B b1{X<1>{}};
};

struct FA : A {
  using A::A; // expected-note 2{{here}}
};
FA fa0{X<0>{}};
FA fa1{X<1>{}}; // expected-error {{calling a protected constructor}}
FA fa2{X<2>{}}; // expected-error {{calling a private constructor}}


// It is deleted if the corresponding constructor [...] is deleted.
struct G {
  G(int) = delete;
  template<typename T> G(T*) = delete;
};
struct H : G {
  using G::G; // expected-note 2{{marked deleted here}}
};
H h1(5); // expected-error {{call to implicitly-deleted function of 'H'}}
H h2("foo"); // expected-error {{call to deleted constructor of 'H'}}


// Core defect: It is also deleted if multiple base constructors generate the
// same signature.
namespace DRnnnn {
  struct A {
    constexpr A(int, float = 0) {}
    explicit A(int, int = 0) {}

    A(int, int, int = 0) = delete;
  };
  struct B : A {
    // FIXME: produce notes indicating why it was deleted
    using A::A; // expected-note {{here}}
  };

  constexpr B b0(0, 0.0f); // ok, constexpr
  B b1(0, 1); // expected-error {{call to implicitly-deleted}}
}
