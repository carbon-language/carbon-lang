// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x

void foo (int operator+); // expected-error{{cannot be the name of a parameter}}
