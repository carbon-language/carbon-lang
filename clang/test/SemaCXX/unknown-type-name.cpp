// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR3990
namespace N {
  struct Wibble {
  };

  typedef Wibble foo;
}
using namespace N;

foo::bar x; // expected-error{{no type named 'bar' in 'N::Wibble'}}

void f() {
  foo::bar  = 4; // expected-error{{no member named 'bar' in 'N::Wibble'}}
}

template<typename T>
struct A {
  typedef T type;
  
  type f();

  type g();
};

template<typename T>
A<T>::type g(T t) { return t; } // expected-error{{missing 'typename'}}

template<typename T>
A<T>::type A<T>::f() { return type(); } // expected-error{{missing 'typename'}}

template<typename T>
void f(int, T::type) { } // expected-error{{missing 'typename'}}

template<typename T>
void f(int, T::type, int) { } // expected-error{{missing 'typename'}}

// FIXME: We know which type specifier should have been specified here. Provide
//        a fix-it to add 'typename A<T>::type'
template<typename T>
A<T>::g() { } // expected-error{{requires a type specifier}}
