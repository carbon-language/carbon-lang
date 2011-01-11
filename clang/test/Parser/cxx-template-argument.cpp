// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T> struct A {};

// Check for template argument lists followed by junk
// FIXME: The diagnostics here aren't great...
A<int+> int x; // expected-error {{expected '>'}} expected-error {{expected unqualified-id}}
A<int x; // expected-error {{expected '>'}}

// PR8912
template <bool> struct S {};
S<bool(2 > 1)> s;
