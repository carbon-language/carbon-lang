// RUN: clang-cc -fsyntax-only -verify %s

class A;

class S {
public:
   template<typename T> struct A { 
     struct Nested {
       typedef T type;
     };
   };
};

int i;
S::A<int>::Nested::type *ip = &i;

template<typename T>
struct X0 {
  template<typename U> void f0(T, U);
  
  template<typename U>
  struct Inner0 {
    void f1(T, U);
  };
};

template<typename X> template<typename Y> void X0<X>::f0(X, Y) { }

// FIXME:
// template<typename X> template<typename Y> void X0<X>::Inner0<Y>::f1(X, Y) { }
