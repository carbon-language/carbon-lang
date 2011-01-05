// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

// If a template-parameter of a class template has a default
// template-argument, each subsequent template-parameter shall either
// have a default template-argument supplied or be a template
// parameter pack.
template<typename> struct vector;

template<typename T = int, typename ...Types> struct X2t;
template<int V = 0, int ...Values> struct X2nt;
template<template<class> class M = vector, template<class> class... Metas>
  struct X2tt;

// If a template-parameter of a primary class template is a template
// parameter pack, it shall be the last template-parameter .
template<typename ...Types, // expected-error{{template parameter pack must be the last template parameter}}
         int After>
struct X0t;

template<int ...Values, // expected-error{{template parameter pack must be the last template parameter}}
         int After>
struct X0nt;

template<template<typename> class ...Templates, // expected-error{{template parameter pack must be the last template parameter}}
         int After>
struct X0tt;

// [ Note: These are not requirements for function templates or class
// template partial specializations because template arguments can be
// deduced (14.8.2). -- end note]
template<typename... Types> struct X1t;
template<typename ...Types, typename T> struct X1t<T, Types...> { };

template<int... Values> struct X1nt;
template<int ...Values, int V> struct X1nt<V, Values...> { };

template<template<int> class... Meta> struct X1tt;
template<template<int> class... Meta, template<int> class M> 
  struct X1tt<M, Meta...> { };

template<typename ...Types, typename T>
void f1t(X1t<T, Types...>);

template<int ...Values, int V>
void f1nt(X1nt<V, Values...>);

template<template<int> class... Meta, template<int> class M> 
void f1tt(X1tt<M, Meta...>);
