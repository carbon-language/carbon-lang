// RUN: %clang_cc1 -fsyntax-only -verify %s

// A type-parameter defines its identifier to be a type-name (if
// declared with class or typename) or template-name (if declared with
// template) in the scope of the template declaration.
template<typename T> struct X0 {
  T* value;
};

template<template<class T> class Y> struct X1 {
  Y<int> value;
};

// [Note: because of the name lookup rules, a template-parameter that
// could be interpreted as either a non-type template-parameter or a
// type-parameter (because its identifier is the name of an already
// existing class) is taken as a type-parameter. For example, 
class T { /* ... */ }; 
int i; 

template<class T, T i> struct X2 {
  void f(T t) 
  { 
    T t1 = i; //template-parameters T and i 
    ::T t2 = ::i; // global namespace members T and i  \
    // expected-error{{cannot initialize}}
  } 
};
