// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

struct NonLit {
  NonLit();
};

struct S {
  static constexpr int a = 0;
  static constexpr int b; // expected-error {{declaration of constexpr variable 'b' requires an initializer}}

  static constexpr int c = 0;
  static const int d;

  static constexpr double e = 0.0; // ok
  static const double f = 0.0; // expected-warning {{extension}}
  static char *const g = 0; // expected-error {{requires 'constexpr' specifier}}
  static const NonLit h = NonLit(); // expected-error {{must be initialized out of line}}
};

constexpr int S::a; // expected-error {{definition of initialized static data member 'a' cannot be marked constexpr}}
constexpr int S::b = 0;

const int S::c;
constexpr int S::d = 0;
