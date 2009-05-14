// RUN: clang-cc -fsyntax-only -verify %s

template<typename T, typename U>
struct X0 {
  void f(T x, U y) { 
    x + y; // expected-error{{invalid operands}}
  }
};

struct X1 { };

template struct X0<int, float>;
template struct X0<int*, int>;
template struct X0<int X1::*, int>; // expected-note{{instantiation of}}

template<typename T>
struct X2 {
  void f(T);

  T g(T x, T y) {
    /* NullStmt */;
  }
};

template struct X2<int>;
