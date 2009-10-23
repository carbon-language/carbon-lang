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
struct Outer {
  template<typename U>
  class Inner0;
  
  template<typename U>
  class Inner1 {
    struct ReallyInner;
    
    T foo(U);
    template<typename V> T bar(V);
    template<typename V> T* bar(V);
    
    static T value1;
    static U value2;
  };
};

template<typename X>
template<typename Y>
class Outer<X>::Inner0 {
public:
  void f(X, Y);
};

template<typename X>
template<typename Y>
void Outer<X>::Inner0<Y>::f(X, Y) {
}

template<typename X>
template<typename Y>
struct Outer<X>::Inner1<Y>::ReallyInner {
  static Y value3;
  
  void g(X, Y);
};

template<typename X>
template<typename Y>
void Outer<X>::Inner1<Y>::ReallyInner::g(X, Y) {
}

template<typename X>
template<typename Y>
X Outer<X>::Inner1<Y>::foo(Y) {
  return X();
}

template<typename X>
template<typename Y>
template<typename Z>
X Outer<X>::Inner1<Y>::bar(Z) {
  return X();
}

template<typename X>
template<typename Y>
template<typename Z>
X* Outer<X>::Inner1<Y>::bar(Z) {
  return 0;
}

template<typename X>
template<typename Y>
X Outer<X>::Inner1<Y>::value1 = 0;

template<typename X>
template<typename Y>
Y Outer<X>::Inner1<Y>::value2 = Y();

template<typename X>
template<typename Y>
Y Outer<X>::Inner1<Y>::ReallyInner::value3 = Y();

template<typename X>
template<typename Y>
Y Outer<X>::Inner1<Y*>::ReallyInner::value4; // expected-error{{Outer<X>::Inner1<Y *>::ReallyInner::}}


template<typename T>
struct X0 { };

template<typename T>
struct X0<T*> {
  template<typename U>
  void f(U u = T()) { }
};

// PR5103
template<typename>
struct X1 {
  template<typename, bool = false> struct B { };
};
template struct X1<int>::B<bool>;
