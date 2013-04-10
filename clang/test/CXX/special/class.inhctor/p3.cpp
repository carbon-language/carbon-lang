// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct B1 {
  B1(int);
  B1(int, int);
};
struct D1 : B1 {
  using B1::B1;
};
D1 d1a(1), d1b(1, 1);

D1 fd1() { return 1; }

struct B2 {
  explicit B2(int, int = 0, int = 0);
};
struct D2 : B2 { // expected-note 2 {{candidate constructor}}
  using B2::B2;
};
D2 d2a(1), d2b(1, 1), d2c(1, 1, 1);

D2 fd2() { return 1; } // expected-error {{no viable conversion}}

struct B3 {
  B3(void*); // expected-note {{inherited from here}}
};
struct D3 : B3 { // expected-note 2 {{candidate constructor}}
  using B3::B3; // expected-note {{candidate constructor (inherited)}}
};
D3 fd3() { return 1; } // expected-error {{no viable conversion}}

template<typename T> struct T1 : B1 {
  using B1::B1;
};
template<typename T> struct T2 : T1<T> {
  using T1<int>::T1;
};
template<typename T> struct T3 : T1<int> {
  using T1<T>::T1;
};
struct U {
  friend T1<int>::T1(int);
  friend T1<int>::T1(int, int);
  friend T2<int>::T2(int);
  friend T2<int>::T2(int, int);
  friend T3<int>::T3(int);
  friend T3<int>::T3(int, int);
};

struct B4 {
  template<typename T> explicit B4(T, int = 0);
};
template<typename T> struct T4 : B4 {
  using B4::B4; // expected-note {{here}}
  template<typename U> T4(U);
};
T4<void> t4a = {0};
T4<void> t4b = {0, 0}; // expected-error {{chosen constructor is explicit}}
