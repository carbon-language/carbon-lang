// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

auto j() -> enum { e3 }; // expected-error{{can not be defined in a type specifier}}
