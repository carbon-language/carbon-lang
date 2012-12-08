// RUN: %clang_cc1 -fsyntax-only -verify %s

//   A program that calls for default-initialization or value-initialization of
//   an entity of reference type is illformed. If T is a cv-qualified type, the
//   cv-unqualified version of T is used for these definitions of
//   zero-initialization, default-initialization, and value-initialization.

struct S { // expected-error {{implicit default constructor for 'S' must explicitly initialize the reference member}}
  int &x; // expected-note {{declared here}} expected-error 3{{reference to type 'int' requires an initializer}}
};
S s; // expected-note {{implicit default constructor for 'S' first required here}}
S f() {
  return S(); // expected-note {{in value-initialization of type 'S' here}}
}

struct T
  : S { // expected-note 2{{in value-initialization of type 'S' here}}
};
T t = T(); // expected-note {{in value-initialization of type 'T' here}}

struct U {
  T t[3]; // expected-note {{in value-initialization of type 'T' here}}
};
U u = U(); // expected-note {{in value-initialization of type 'U' here}}

// Ensure that we handle C++11 in-class initializers properly as an extension.
// In this case, there is no user-declared default constructor, so we
// recursively apply the value-initialization checks, but we will emit a
// constructor call anyway, because the default constructor is not trivial.
struct V {
  int n;
  int &r = n; // expected-warning {{C++11}}
};
V v = V(); // ok
struct W {
  int n;
  S s = { n }; // expected-warning {{C++11}}
};
W w = W(); // ok

// Ensure we're not faking this up by making the default constructor
// non-trivial.
#define static_assert(B, S) typedef int assert_failed[(B) ? 1 : -1];
static_assert(__has_trivial_constructor(S), "");
static_assert(__has_trivial_constructor(T), "");
static_assert(__has_trivial_constructor(U), "");
static_assert(!__has_trivial_constructor(V), "");
static_assert(!__has_trivial_constructor(W), "");
