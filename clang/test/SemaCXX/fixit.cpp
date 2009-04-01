// RUN: clang-cc -fsyntax-only -pedantic -verify %s 

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. Eventually,
   we would like to actually try to perform the suggested
   modifications and compile the result to test that no warnings
   remain. */

struct C1 { };
struct C2 : virtual public virtual C1 { }; // expected-error{{duplicate}}

template<int Value> struct CT { };

CT<10 >> 2> ct; // expected-warning{{require parentheses}}
