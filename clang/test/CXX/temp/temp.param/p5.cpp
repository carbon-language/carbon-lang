// RUN: %clang_cc1 -verify %s -std=c++11

template<const int I> struct S { // expected-note {{instantiation}}
  decltype(I) n;
  int &&r = I; // expected-warning 2{{binding reference member 'r' to a temporary value}} expected-note 2{{declared here}}
};
S<5> s;

template<typename T, T v> struct U { // expected-note {{instantiation}}
  decltype(v) n;
  int &&r = v; // expected-warning {{binding reference member 'r' to a temporary value}} expected-note {{declared here}}
};
U<const int, 6> u;
