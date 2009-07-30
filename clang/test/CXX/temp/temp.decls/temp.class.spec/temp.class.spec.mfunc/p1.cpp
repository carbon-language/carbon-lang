// RUN: clang-cc -fsyntax-only -verify %s

template<typename T, int N>
struct A;

template<typename T>
struct A<T*, 2> {
  void f(T*);
  
  static T value;
};

template<class X> void A<X*, 2>::f(X*) { }

template<class X> X A<X*, 2>::value;
