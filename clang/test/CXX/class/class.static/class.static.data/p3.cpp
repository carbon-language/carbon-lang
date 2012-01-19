// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct NonLit {
  NonLit();
};

struct S {
  static constexpr int a = 0;
  static constexpr int b; // expected-error {{declaration of constexpr static data member 'b' requires an initializer}}

  static constexpr int c = 0;
  static const int d;
  static const int d2 = 0;

  static constexpr double e = 0.0; // ok
  static const double f = 0.0; // expected-warning {{extension}} expected-note {{use 'constexpr' specifier}}
  static char *const g = 0; // expected-error {{requires 'constexpr' specifier}}
  static const NonLit h = NonLit(); // expected-error {{must be initialized out of line}}
};

constexpr int S::a;
constexpr int S::b = 0;

const int S::c;
constexpr int S::d = 0;
constexpr int S::d2;

template<typename T>
struct U {
  static constexpr int a = 0;
  static constexpr int b; // expected-error {{declaration of constexpr static data member 'b' requires an initializer}}
  // FIXME: It'd be nice to error on this at template definition time.
  static constexpr NonLit h = NonLit(); // expected-error 2{{must be initialized by a constant expression}} expected-note 2{{non-literal type}}
  static constexpr T c = T(); // expected-error {{must be initialized by a constant expression}} expected-note {{non-literal type}}
};

U<int> u1; // expected-note {{here}}
U<NonLit> u2; // expected-note {{here}}

static_assert(U<int>::a == 0, "");
