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
