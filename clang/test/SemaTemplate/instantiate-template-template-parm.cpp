// RUN: %clang_cc1 -fsyntax-only -verify %s
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
apply<add_reference, float>::type fr = i; // expected-error{{non-const lvalue reference to type 'float' cannot bind to a value of unrelated type 'int'}}

// Template template parameters
template<int> struct B; // expected-note{{has a different type 'int'}}

template<typename T, 
         template<T Value> class X> // expected-error{{cannot have type 'float'}} \
                                    // expected-note{{with type 'long'}}
struct X0 { };

X0<int, B> x0b1;
X0<float, B> x0b2; // expected-note{{while substituting}}
X0<long, B> x0b3; // expected-error{{template template argument has different template parameters}}

template<template<int V> class TT> // expected-note{{parameter with type 'int'}}
struct X1 { };

template<typename T, template<T V> class TT>
struct X2 {
  X1<TT> x1; // expected-error{{has different template parameters}}
};

template<int V> struct X3i { };
template<long V> struct X3l { }; // expected-note{{different type 'long'}}

X2<int, X3i> x2okay;
X2<long, X3l> x2bad; // expected-note{{instantiation}}
