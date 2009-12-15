// RUN: %clang_cc1 -fsyntax-only -std=c++0x -verify %s
template<typename T> struct X;
template<int I> struct Y;

X<X<int>> *x1;

Y<(1 >> 2)> *y1;
Y<1 >> 2> *y2; // FIXME: expected-error{{expected unqualified-id}}

X<X<X<X<X<int>>>>> *x2;

template<> struct X<int> { };
typedef X<int> X_int;
struct Z : X_int { };

void f(const X<int> x) {
  (void)reinterpret_cast<X<int>>(x); // expected-error{{reinterpret_cast from}}
  (void)reinterpret_cast<X<X<X<int>>>>(x); // expected-error{{reinterpret_cast from}}

  X<X<int>> *x1;
}

