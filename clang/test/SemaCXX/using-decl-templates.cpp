// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T> struct A {
  void f() { }
  struct N { }; // expected-note{{target of using declaration}}
};

template<typename T> struct B : A<T> {
  using A<T>::f;
  using A<T>::N; // expected-error{{dependent using declaration resolved to type without 'typename'}}
  
  using A<T>::foo; // expected-error{{no member named 'foo'}}
  using A<double>::f; // expected-error{{using declaration refers into 'A<double>::', which is not a base class of 'B<int>'}}
};

B<int> a; // expected-note{{in instantiation of template class 'B<int>' requested here}}

template<typename T> struct C : A<T> {
  using A<T>::f;
  
  void f() { };
};

template <typename T> struct D : A<T> {
  using A<T>::f;
  
  void f();
};

template<typename T> void D<T>::f() { }

template<typename T> struct E : A<T> {
  using A<T>::f;

  void g() { f(); }
};

namespace test0 {
  struct Base {
    int foo;
  };
  template<typename T> struct E : Base {
    using Base::foo;
  };

  template struct E<int>;
}
