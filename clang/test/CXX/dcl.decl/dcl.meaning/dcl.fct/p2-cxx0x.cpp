// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s 

auto a() -> int; // ok
const auto b() -> int; // expected-error {{function with trailing return type must specify return type 'auto', not 'auto const'}}
auto *c() -> int; // expected-error {{function with trailing return type must specify return type 'auto', not 'auto *'}}
