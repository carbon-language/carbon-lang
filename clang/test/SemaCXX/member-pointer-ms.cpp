// RUN: %clang_cc1 -cxx-abi microsoft -fsyntax-only -verify %s

// Test that we reject pointers to members of incomplete classes (for now)
struct A; //expected-note{{forward declaration of 'A'}}
int A::*pai1; //expected-error{{incomplete type 'A'}}

