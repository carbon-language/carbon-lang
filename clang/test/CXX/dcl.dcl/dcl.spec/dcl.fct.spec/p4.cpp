// RUN: %clang_cc1 -fsyntax-only -verify %s

void f0() { // expected-note {{previous definition is here}}
}

inline void f0(); // expected-error {{inline declaration of 'f0' follows non-inline definition}}
