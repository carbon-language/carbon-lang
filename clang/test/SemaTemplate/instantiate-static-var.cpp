// RUN: clang-cc -fsyntax-only -verify %s

template<typename T, T Divisor>
class X {
public:
  static const T value = 10 / Divisor; // expected-error{{in-class initializer is not an integral constant expression}}
};

int array1[X<int, 2>::value == 5? 1 : -1];
X<int, 0> xi0; // expected-note{{in instantiation of template class 'class X<int, 0>' requested here}}


template<typename T>
class Y {
  static const T value = 0; // expected-error{{'value' can only be initialized if it is a static const integral data member}}
};

Y<float> fy; // expected-note{{in instantiation of template class 'class Y<float>' requested here}}
