// RUN: clang-cc -verify %s
// XFAIL

void f0() {
}

inline void f0(); // expected-error {{function definition cannot preceed inline declaration}}
