// RUN: %clang_cc1 -verify %s

// expected-no-diagnostics

struct A { int n; };
struct B { float n; };
struct C : A, B {};
struct D : virtual C {};
struct E : virtual C { char n; };
struct F : D, E {} f;
char &k = f.n;
