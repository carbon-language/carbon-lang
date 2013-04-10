// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// Straight from the standard
struct B1 {
  B1(int); // expected-note {{previous constructor}} expected-note {{conflicting constructor}}
};
struct B2 {
  B2(int); // expected-note {{conflicting constructor}}
};
struct D1 : B1, B2 {
  using B1::B1; // expected-note {{inherited here}}
  using B2::B2; // expected-error {{already inherited constructor with the same signature}}
};
struct D2 : B1, B2 {
  using B1::B1;
  using B2::B2;
  D2(int);
};

template<typename T> struct B3 {
  B3(T); // expected-note {{previous constructor}}
};
template<typename T> struct B4 : B3<T>, B1 {
  B4();
  using B3<T>::B3; // expected-note {{inherited here}}
  using B1::B1; // expected-error {{already inherited}}
};
B4<char> b4c;
B4<int> b4i; // expected-note {{here}}

struct B5 {
  template<typename T> B5(T); // expected-note {{previous constructor}}
};
struct B6 {
  template<typename T> B6(T); // expected-note {{conflicting constructor}}
};
struct B7 {
  template<typename T, int> B7(T);
};
struct D56 : B5, B6, B7 {
  using B5::B5; // expected-note {{inherited here}}
  using B6::B6; // expected-error {{already inherited}}
};
struct D57 : B5, B6, B7 {
  using B5::B5;
  using B7::B7; // ok, not the same signature
};
