// RUN: clang-cc -fsyntax-only -verify %s

template<template<typename T> class MetaFun, typename Value>
struct apply {
  typedef typename MetaFun<Value>::type type;
};

template<class T>
struct add_pointer {
  typedef T* type;
};

template<class T>
struct add_reference {
  typedef T& type;
};

int i;
apply<add_pointer, int>::type ip = &i;
apply<add_reference, int>::type ir = i;
apply<add_reference, float>::type fr = i; // expected-error{{non-const lvalue reference to type 'float' cannot be initialized with a value of type 'int'}}
