// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
template<typename T>
class X0 {
  friend T;
};

class Y1 { };
enum E1 { };
X0<Y1> x0a;
X0<Y1 *> x0b;
X0<int> x0c;
X0<E1> x0d;

template<typename T>
class X1 {
  friend typename T::type; // expected-error{{no type named 'type' in 'Y1'}}
};

struct Y2 {
  struct type { };
};

struct Y3 {
  typedef int type;
};

X1<Y2> x1a;
X1<Y3> x1b;
X1<Y1> x1c; // expected-note{{in instantiation of template class 'X1<Y1>' requested here}}
