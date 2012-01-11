// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// If the implicitly-defined constructor would satisfy the requirements of a
// constexpr constructor, the implicitly-defined constructor is constexpr.
struct Constexpr1 {
  constexpr Constexpr1() : n(0) {}
  int n;
};
constexpr Constexpr1 c1a = Constexpr1(Constexpr1()); // ok
constexpr Constexpr1 c1b = Constexpr1(Constexpr1(c1a)); // ok

struct Constexpr2 {
  Constexpr1 ce1;
  constexpr Constexpr2() = default;
  constexpr Constexpr2(const Constexpr2 &o) : ce1(o.ce1) {}
  // no move constructor
};

constexpr Constexpr2 c2a = Constexpr2(Constexpr2()); // ok
constexpr Constexpr2 c2b = Constexpr2(Constexpr2(c2a)); // ok

struct Constexpr3 {
  Constexpr2 ce2;
  // all special constructors are constexpr, move ctor calls ce2's copy ctor
};

constexpr Constexpr3 c3a = Constexpr3(Constexpr3()); // ok
constexpr Constexpr3 c3b = Constexpr3(Constexpr3(c3a)); // ok

struct NonConstexprCopy {
  constexpr NonConstexprCopy() = default;
  NonConstexprCopy(const NonConstexprCopy &);
  constexpr NonConstexprCopy(NonConstexprCopy &&) = default;

  int n = 42;
};

NonConstexprCopy::NonConstexprCopy(const NonConstexprCopy &) = default; // expected-note {{here}}

constexpr NonConstexprCopy ncc1 = NonConstexprCopy(NonConstexprCopy()); // ok
constexpr NonConstexprCopy ncc2 = ncc1; // expected-error {{constant expression}} expected-note {{non-constexpr constructor}}

struct NonConstexprDefault {
  NonConstexprDefault() = default;
  constexpr NonConstexprDefault(int n) : n(n) {}
  int n;
};
struct Constexpr4 {
  NonConstexprDefault ncd;
};

constexpr NonConstexprDefault ncd = NonConstexprDefault(NonConstexprDefault(1));
constexpr Constexpr4 c4a = { ncd };
constexpr Constexpr4 c4b = Constexpr4(c4a);
constexpr Constexpr4 c4c = Constexpr4(static_cast<Constexpr4&&>(const_cast<Constexpr4&>(c4b)));

struct Constexpr5Base {};
struct Constexpr5 : Constexpr5Base { constexpr Constexpr5() {} };
constexpr Constexpr5 ce5move = Constexpr5();
constexpr Constexpr5 ce5copy = ce5move;
