// RUN: clang-cc -fsyntax-only -verify %s
class A { 
  void f(A* p = this) { }	// expected-error{{invalid use of 'this'}}
};
