// RUN: clang-cc -fsyntax-only -verify %s

x; // expected-error{{C++ requires a type specifier for all declarations}}

f(int y) { return y; } // expected-error{{C++ requires a type specifier for all declarations}}
