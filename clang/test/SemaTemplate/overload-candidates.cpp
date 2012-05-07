// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
const T& min(const T&, const T&); // expected-note{{candidate template ignored: deduced conflicting types for parameter 'T' ('int' vs. 'long')}}

void test_min() {
  (void)min(1, 2l); // expected-error{{no matching function for call to 'min'}}
}

template<typename R, typename T>
R *dyn_cast(const T&); // expected-note{{candidate template ignored: couldn't infer template argument 'R'}}

void test_dyn_cast(int* ptr) {
  (void)dyn_cast(ptr); // expected-error{{no matching function for call to 'dyn_cast'}}
}

template<int I, typename T> 
  void get(const T&); // expected-note{{candidate template ignored: invalid explicitly-specified argument for template parameter 'I'}}
template<template<class T> class, typename T> 
  void get(const T&); // expected-note{{candidate template ignored: invalid explicitly-specified argument for 1st template parameter}}

void test_get(void *ptr) {
  get<int>(ptr); // expected-error{{no matching function for call to 'get'}}
}

template<typename T>
  typename T::type get_type(const T&); // expected-note{{candidate template ignored: substitution failure [with T = int *]: type 'int *' cannot be used prior to '::'}}
template<typename T>
  void get_type(T *, int[(int)sizeof(T) - 9] = 0); // expected-note{{candidate template ignored: substitution failure [with T = int]: array size is negative}}

void test_get_type(int *ptr) {
  (void)get_type(ptr); // expected-error{{no matching function for call to 'get_type'}}
}

struct X {
  template<typename T>
  const T& min(const T&, const T&); // expected-note{{candidate template ignored: deduced conflicting types for parameter 'T' ('int' vs. 'long')}}
};

void test_X_min(X x) {
  (void)x.min(1, 2l); // expected-error{{no matching member function for call to 'min'}}
}
