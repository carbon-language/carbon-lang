// RUN: %clang_cc1 -std=c++14 -fconcepts-ts -x c++ %s -verify

// Test parsing of constraint-expressions in cases where the grammar is
// ambiguous with the expectation that the longest token sequence which matches
// the syntax is consumed without backtracking.

// type-specifier-seq in conversion-type-id
template <typename T> requires (bool)&T::operator short
unsigned int foo(); // expected-error {{C++ requires a type specifier for all declarations}}

// type-specifier-seq in new-type-id
template <typename T> requires (bool)sizeof new (T::f()) short
unsigned int bar(); // expected-error {{C++ requires a type specifier for all declarations}}

template<typename T> requires (bool)sizeof new (T::f()) unsigned // expected-error {{'struct' cannot be signed or unsigned}}
struct X { }; // expected-error {{'X' cannot be defined in a type specifier}}

// C-style cast
// of function call on function-style cast
template <typename T> requires (bool(T()))
T (*fp)(); // expected-error {{use of undeclared identifier 'fp'}}

// function-style cast
// as the callee in a function call
struct A {
  static int t;
  template <typename T> requires bool(T())
  (A(T (&t))) { } // expected-error {{called object type 'bool' is not a function or function pointer}}
};
