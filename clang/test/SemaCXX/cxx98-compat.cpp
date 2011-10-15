// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wc++98-compat -verify %s

template<typename ...T>  // expected-warning {{variadic templates are incompatible with C++98}}
class Variadic1 {};

template<template<typename> class ...T>  // expected-warning {{variadic templates are incompatible with C++98}}
class Variadic2 {};

template<int ...I>  // expected-warning {{variadic templates are incompatible with C++98}}
class Variadic3 {};

int alignas(8) with_alignas; // expected-warning {{'alignas' is incompatible with C++98}}
int with_attribute [[ ]]; // expected-warning {{attributes are incompatible with C++98}}

void Literals() {
  (void)u8"str"; // expected-warning {{unicode literals are incompatible with C++98}}
  (void)u"str"; // expected-warning {{unicode literals are incompatible with C++98}}
  (void)U"str"; // expected-warning {{unicode literals are incompatible with C++98}}
  (void)u'x'; // expected-warning {{unicode literals are incompatible with C++98}}
  (void)U'x'; // expected-warning {{unicode literals are incompatible with C++98}}

  (void)u8R"X(str)X"; // expected-warning {{raw string literals are incompatible with C++98}}
  (void)uR"X(str)X"; // expected-warning {{raw string literals are incompatible with C++98}}
  (void)UR"X(str)X"; // expected-warning {{raw string literals are incompatible with C++98}}
  (void)R"X(str)X"; // expected-warning {{raw string literals are incompatible with C++98}}
  (void)LR"X(str)X"; // expected-warning {{raw string literals are incompatible with C++98}}
}

template<typename T> struct S {};
S<::S<void> > s; // expected-warning {{'<::' is treated as digraph '<:' (aka '[') followed by ':' in C++98}}
