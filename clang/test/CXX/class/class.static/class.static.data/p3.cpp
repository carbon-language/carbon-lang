// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct NonLit {
  NonLit();
};

struct S {
  static constexpr int a = 0;
  static constexpr int b; // expected-error {{declaration of constexpr variable 'b' requires an initializer}}

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
