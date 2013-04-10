// RUN: %clang_cc1 -std=c++11 -verify %s

template<int> struct X {};

// Constructor characteristics are:
//   - the template parameter list
//   - the parameter-type-list
//   - absence or presence of explicit
//   - absence or presence of constexpr
struct A {
  A(X<0>) {} // expected-note 2{{here}}
  constexpr A(X<1>) {}
  explicit A(X<2>) {} // expected-note 3{{here}}
  explicit constexpr A(X<3>) {} // expected-note 2{{here}}
};

A a0 { X<0>{} };
A a0i = { X<0>{} };
constexpr A a0c { X<0>{} }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}
constexpr A a0ic = { X<0>{} }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}

A a1 { X<1>{} };
A a1i = { X<1>{} };
constexpr A a1c { X<1>{} };
constexpr A a1ic = { X<1>{} };

A a2 { X<2>{} };
A a2i = { X<2>{} }; // expected-error {{constructor is explicit}}
constexpr A a2c { X<2>{} }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}
constexpr A a2ic = { X<2>{} }; // expected-error {{constructor is explicit}}

A a3 { X<3>{} };
A a3i = { X<3>{} }; // expected-error {{constructor is explicit}}
constexpr A a3c { X<3>{} };
constexpr A a3ic = { X<3>{} }; // expected-error {{constructor is explicit}}


struct B : A {
  using A::A; // expected-note 7{{here}}
};

B b0 { X<0>{} };
B b0i = { X<0>{} };
constexpr B b0c { X<0>{} }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}
constexpr B b0ic = { X<0>{} }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}

B b1 { X<1>{} };
B b1i = { X<1>{} };
constexpr B b1c { X<1>{} };
constexpr B b1ic = { X<1>{} };

B b2 { X<2>{} };
B b2i = { X<2>{} }; // expected-error {{constructor is explicit}}
constexpr B b2c { X<2>{} }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}
constexpr B b2ic = { X<2>{} }; // expected-error {{constructor is explicit}}

B b3 { X<3>{} };
B b3i = { X<3>{} }; // expected-error {{constructor is explicit}}
constexpr B b3c { X<3>{} };
constexpr B b3ic = { X<3>{} }; // expected-error {{constructor is explicit}}


// 'constexpr' is OK even if the constructor doesn't obey the constraints.
struct NonLiteral { NonLiteral(); };
struct NonConstexpr { NonConstexpr(); constexpr NonConstexpr(int); }; // expected-note {{here}}
struct Constexpr { constexpr Constexpr(int) {} };

struct BothNonLiteral : NonLiteral, Constexpr { using Constexpr::Constexpr; }; // expected-note {{base class 'NonLiteral' of non-literal type}}
constexpr BothNonLiteral bothNL{42}; // expected-error {{constexpr variable cannot have non-literal type 'const BothNonLiteral'}}

struct BothNonConstexpr : NonConstexpr, Constexpr { using Constexpr::Constexpr; }; // expected-note {{non-constexpr constructor 'NonConstexpr}}
constexpr BothNonConstexpr bothNC{42}; // expected-error {{must be initialized by a constant expression}} expected-note {{in call to 'BothNonConstexpr(42)'}}


struct ConstexprEval {
  constexpr ConstexprEval(int a, const char *p) : k(p[a]) {}
  char k;
};
struct ConstexprEval2 {
  char k2 = 'x';
};
struct ConstexprEval3 : ConstexprEval, ConstexprEval2 {
  using ConstexprEval::ConstexprEval;
};
constexpr ConstexprEval3 ce{4, "foobar"};
static_assert(ce.k == 'a', "");
static_assert(ce.k2 == 'x', "");


struct TemplateCtors {
  constexpr TemplateCtors() {}
  template<template<int> class T> TemplateCtors(X<0>, T<0>);
  template<int N> TemplateCtors(X<1>, X<N>);
  template<typename T> TemplateCtors(X<2>, T);

  template<typename T = int> TemplateCtors(int, int = 0, int = 0); // expected-note {{inherited from here}}
};

struct UsingTemplateCtors : TemplateCtors {
  using TemplateCtors::TemplateCtors; // expected-note 4{{here}} expected-note {{candidate}}

  constexpr UsingTemplateCtors(X<0>, X<0>) {}
  constexpr UsingTemplateCtors(X<1>, X<1>) {}
  constexpr UsingTemplateCtors(X<2>, X<2>) {}

  template<int = 0> constexpr UsingTemplateCtors(int) {} // expected-note {{candidate}}
  template<typename T = void> constexpr UsingTemplateCtors(int, int) {}
  template<typename T, typename U> constexpr UsingTemplateCtors(int, int, int) {}
};

template<int> struct Y {};
constexpr UsingTemplateCtors uct1{ X<0>{}, X<0>{} };
constexpr UsingTemplateCtors uct2{ X<0>{}, Y<0>{} }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}
constexpr UsingTemplateCtors uct3{ X<1>{}, X<0>{} }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}
constexpr UsingTemplateCtors uct4{ X<1>{}, X<1>{} };
constexpr UsingTemplateCtors uct5{ X<2>{}, 0 }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}
constexpr UsingTemplateCtors uct6{ X<2>{}, X<2>{} };

constexpr UsingTemplateCtors utc7{ 0 }; // expected-error {{ambiguous}}
constexpr UsingTemplateCtors utc8{ 0, 0 }; // ok
constexpr UsingTemplateCtors utc9{ 0, 0, 0 }; // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr}}
