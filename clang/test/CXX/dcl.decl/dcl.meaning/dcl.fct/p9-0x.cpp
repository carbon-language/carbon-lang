// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

auto j() -> enum { e3 }; // expected-error{{unnamed enumeration must be a definition}} expected-error {{requires a specifier or qualifier}} expected-error {{without trailing return type}}
