// RUN: %clang_cc1 -std=c++11 -verify %s

struct A {
  constexpr A(const int&) : rval(false) {}
  constexpr A(const int&&) : rval(true) {}
  bool rval;
};
struct B : A {
  using A::A;
};

constexpr int k = 0;
constexpr A a0{0};
constexpr A a1{k};
constexpr B b0{0};
// This performs static_cast<(const int&)&&>(k), so calls the A(const int&)
// constructor.
constexpr B b1{k};

static_assert(a0.rval && !a1.rval && b0.rval && !b1.rval, "");

struct C {
  template<typename T> constexpr C(T t) : v(t) {}
  int v;
};
struct D : C {
  using C::C;
};
static_assert(D(123).v == 123, "");

// FIXME: This diagnostic sucks.
template<typename T> constexpr D::D(T t) : C(t) {} // expected-error {{definition of implicitly declared function}}
