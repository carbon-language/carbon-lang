// RUN: %clang_cc1 -std=c++1y %s -verify -pedantic-errors

// FIXME: many diagnostics here say 'variably modified type'.
//        catch this case and say 'array of runtime bound' instead.

namespace std { struct type_info; }

struct S {
  int arr[__SIZE_MAX__ / 32];
};
S s[32]; // expected-error {{array is too large}}

int n;
int a[n]; // expected-error {{not allowed at file scope}}

struct T {
  int a[n]; // expected-error {{fields must have a constant size}}
  static int b[n]; // expected-error {{not allowed at file scope}}
};

int g(int n, int a[n]);

template<typename T> struct X {};
template<int N, int[N]> struct Y {};
template<int[n]> struct Z {}; // expected-error {{of variably modified type}}

int f(int n) {
  int arb[n]; // expected-note 3{{here}}
  [arb] {} (); // expected-error {{cannot be captured}}

  // FIXME: an array of runtime bound can be captured by reference.
  [&arb] { // expected-error {{cannot be captured}}
    // Capturing the array implicitly captures the bound, if we need it
    // in a range-based for loop.
    for (auto &n : arb) { } // expected-error {{cannot be captured}}
  } ();

  X<int[n]> x; // expected-error {{variably modified type}}

  int arb_neg[-1]; // expected-error {{negative size}}
  int arb_of_array[n][2];
  int arr[3] = { 1, 2, 3, 4 }; // expected-error {{excess elements}}
  char foo[4] = "fool"; // expected-error {{initializer-string for char array is too long}}

  static int not_auto1[n]; // expected-error {{can not have 'static'}}
  extern int not_auto2[n]; // expected-error {{can not have 'extern'}}
  // FIXME: say 'thread_local' not 'static'.
  thread_local int not_auto1[n]; // expected-error {{can not have 'static'}}

  // FIXME: these should all be invalid.
  auto &&ti1 = typeid(arb);
  auto &&ti2 = typeid(int[n]);
  auto &&so1 = sizeof(arb);
  auto &&so2 = sizeof(int[n]);
  auto *p = &arb;
  decltype(arb) arb2;
  int (*arbp)[n] = 0;
  const int (&arbr)[n] = arbr; // expected-warning {{not yet bound}}
  typedef int arbty[n];
  int array_of_arb[2][n];

  struct Dyn { Dyn() {} Dyn(int) {} ~Dyn() {} };

  // FIXME: these should be valid.
  int arb_dynamic[n] = { 1, 2, 3, 4 }; // expected-error {{may not be initialized}}
  Dyn dyn[n]; // expected-error {{non-POD}}
  Dyn dyn_init[n] = { 1, 2, 3, 4 }; // expected-error {{non-POD}}
}
