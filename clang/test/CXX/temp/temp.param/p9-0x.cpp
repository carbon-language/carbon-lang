// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

// A default template-argument may be specified for any kind of
// template-parameter that is not a template parameter pack.
template<typename ...Types = int> // expected-error{{template parameter pack cannot have a default argument}}
struct X0;

template<int ...Values = 0> // expected-error{{template parameter pack cannot have a default argument}}
struct X1;

template<typename T> struct vector;

template<template<class> class ...Templates = vector> // expected-error{{template parameter pack cannot have a default argument}}
struct X2; 
