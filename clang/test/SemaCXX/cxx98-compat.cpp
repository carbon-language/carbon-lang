// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wc++98-compat -verify %s

template<typename ...T>  // expected-warning {{variadic templates are incompatible with C++98}}
class Variadic1 {};

template<template<typename> class ...T>  // expected-warning {{variadic templates are incompatible with C++98}}
class Variadic2 {};

template<int ...I>  // expected-warning {{variadic templates are incompatible with C++98}}
class Variadic3 {};

int alignas(8) with_alignas; // expected-warning {{'alignas' is incompatible with C++98}}
int with_attribute [[ ]]; // expected-warning {{attributes are incompatible with C++98}}
