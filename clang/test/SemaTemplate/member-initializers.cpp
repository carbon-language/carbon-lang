// RUN: clang-cc -fsyntax-only -verify %s

template<typename T> struct A {
  A() : j(10), i(10) { }
  
  int i;
  int j;
};

template<typename T> struct B : A<T> {
  B() : A<T>() { }
};

