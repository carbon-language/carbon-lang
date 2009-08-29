// RUN: clang-cc -fsyntax-only -verify %s

template<typename T> struct A {
  void f() { }
  struct N { };
};

template<typename T> struct B : A<T> {
  using A<T>::f;
  using A<T>::N;
  
  using A<T>::foo; // expected-error{{no member named 'foo'}}
  using A<double>::f; // expected-error{{using declaration refers into 'A<double>::', which is not a base class of 'B'}}
};

B<int> a; // expected-note{{in instantiation of template class 'struct B<int>' requested here}}

template<typename T> struct C : A<T> {
  using A<T>::f;
  
  void f() { };
};

template <typename T> struct D : A<T> {
  using A<T>::f;
  
  void f();
};

template<typename T> void D<T>::f() { }
