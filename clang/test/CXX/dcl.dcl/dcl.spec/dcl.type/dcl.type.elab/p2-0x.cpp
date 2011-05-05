// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

struct A { };
template<typename T> using X = A; // expected-note {{declared here}}
struct X<int>* p2; // expected-error {{elaborated type refers to a type alias template}}


template<typename T> using Id = T; // expected-note {{declared here}}
template<template<typename> class F>
struct Y {
  struct F<int> i; // expected-error {{elaborated type refers to a type alias template}}
};
template struct Y<Id>; // expected-note {{requested here}}
