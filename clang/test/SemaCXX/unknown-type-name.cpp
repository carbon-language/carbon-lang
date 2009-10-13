// RUN: clang-cc -fsyntax-only -verify %s

// PR3990
namespace N {
  struct Wibble {
  };

  typedef Wibble foo;
}
using namespace N;

foo::bar x; // expected-error{{no type named 'bar' in 'struct N::Wibble'}}

void f() {
  foo::bar  = 4; // expected-error{{no member named 'bar' in 'struct N::Wibble'}}
}

template<typename T>
struct A {
  typedef T type;
  
  type f();
};

template<typename T>
A<T>::type g(T t) { return t; } // expected-error{{missing 'typename'}}

template<typename T>
A<T>::type A<T>::f() { return type(); } // expected-error{{missing 'typename'}}
