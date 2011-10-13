// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// A constexpr specifier used in an object declaration declares the object as
// const.
constexpr int a = 0;
extern const int a;

int i;
constexpr int *b = &i;
extern int *const b;

constexpr int &c = i;
extern int &c;

constexpr int (*d)(int) = 0;
extern int (*const d)(int);

// A variable declaration which uses the constexpr specifier shall have an
// initializer and shall be initialized by a constant expression.
constexpr int ni1; // expected-error {{declaration of constexpr variable 'ni1' requires an initializer}}
constexpr struct C { C(); } ni2; // expected-error {{declaration of constexpr variable 'ni2' requires an initializer}}
constexpr double &ni3; // expected-error {{declaration of constexpr variable 'ni3' requires an initializer}}

constexpr int nc1 = i; // expected-error {{constexpr variable 'nc1' must be initialized by a constant expression}}
constexpr C nc2 = C(); // expected-error {{constexpr variable 'nc2' must be initialized by a constant expression}}
int &f();
constexpr int &nc3 = f(); // expected-error {{constexpr variable 'nc3' must be initialized by a constant expression}}
constexpr int nc4(i); // expected-error {{constexpr variable 'nc4' must be initialized by a constant expression}}
constexpr C nc5((C())); // expected-error {{constexpr variable 'nc5' must be initialized by a constant expression}}
int &f();
constexpr int &nc6(f()); // expected-error {{constexpr variable 'nc6' must be initialized by a constant expression}}

struct pixel {
  int x, y;
};
constexpr pixel ur = { 1294, 1024 }; // ok
constexpr pixel origin;              // expected-error {{requires an initializer}}
