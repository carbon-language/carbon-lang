// RUN: clang-cc -fsyntax-only -verify %s

namespace PR5557 {
template <class T> struct A {
  A();
  virtual int a(T x);
};
template<class T> A<T>::A() {}
template<class T> int A<T>::a(T x) { 
  return *x; // expected-error{{requires pointer operand}}
}

A<int> x; // expected-note{{instantiation}}

template<typename T>
struct X {
  virtual void f();
};

template<>
void X<int>::f() { }
}
