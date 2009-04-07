// RUN: clang-cc -fsyntax-only -verify %s

// Check that we don't allow illegal uses of inline
inline int a; // expected-error{{'inline' can only appear on functions}}
typedef inline int b; // expected-error{{'inline' can only appear on functions}}
int d(inline int a); // expected-error{{'inline' can only appear on functions}}
