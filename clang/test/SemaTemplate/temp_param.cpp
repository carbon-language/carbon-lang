// RUN: clang-cc -fsyntax-only -verify %s 

class X;

// C++ [temp.param]p4
typedef int INT;
enum E { enum1, enum2 };
template<int N> struct A1;
template<INT N, INT M> struct A2;
template<enum E x, E y> struct A3;
template<int &X> struct A4;
template<int *Ptr> struct A5;
template<int (&f)(int, int)> struct A6;
template<int (*fp)(float, double)> struct A7;
template<int X::*pm> struct A8;
template<float (X::*pmf)(float, int)> struct A9;
template<typename T, T x> struct A10;

template<float f> struct A11; // expected-error{{a non-type template parameter cannot have type 'float'}}

template<void *Ptr> struct A12; // expected-error{{a non-type template parameter cannot have type 'void *'}}

// C++ [temp.param]p8
template<int X[10]> struct A5;
template<int f(float, double)> struct A7;

// C++ [temp.param]p11:
template<typename> struct Y1; // expected-note{{too few template parameters in template template argument}}
template<typename, int> struct Y2;

template<class T1 = int, // expected-note{{previous default template argument defined here}}
         class T2>  // expected-error{{template parameter missing a default argument}}
  class B1;

template<template<class> class = Y1, // expected-note{{previous default template argument defined here}}
         template<class> class> // expected-error{{template parameter missing a default argument}}
  class B1t;

template<int N = 5,  // expected-note{{previous default template argument defined here}}
         int M>  // expected-error{{template parameter missing a default argument}}
  class B1n;

// Check for bogus template parameter shadow warning.
template<template<class T> class,
         template<class T> class>
  class B1noshadow;

// C++ [temp.param]p10:
template<class T1, class T2 = int> class B2; 
template<class T1 = int, class T2> class B2;

template<template<class, int> class, template<class> class = Y1> class B2t;
template<template<class, int> class = Y2, template<class> class> class B2t;

template<int N, int M = 5> class B2n;
template<int N = 5, int M> class B2n;

// C++ [temp.param]p12:
template<class T1, 
         class T2 = int> // expected-note{{previous default template argument defined here}}
  class B3;
template<class T1, typename T2> class B3;
template<class T1, 
         typename T2 = float> // expected-error{{template parameter redefines default argument}}
  class B3;

template<template<class, int> class, 
         template<class> class = Y1> // expected-note{{previous default template argument defined here}}
  class B3t;

template<template<class, int> class, template<class> class> class B3t;

template<template<class, int> class, 
         template<class> class = Y1> // expected-error{{template parameter redefines default argument}}
  class B3t;

template<int N, 
         int M = 5> // expected-note{{previous default template argument defined here}}
  class B3n;

template<int N, int M> class B3n;

template<int N, 
         int M = 7>  // expected-error{{template parameter redefines default argument}}
  class B3n;

// Check validity of default arguments
template<template<class, int> class // expected-note{{previous template template parameter is here}}
           = Y1> // expected-error{{template template argument has different template parameters than its corresponding template template parameter}}
  class C1; 
