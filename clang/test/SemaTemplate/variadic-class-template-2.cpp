// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x

// Type parameters packs
template <typename ...> struct TS1 {}; // expected-note{{template parameter is declared here}}
template struct TS1<>;
template struct TS1<int>;
template struct TS1<int, int>;
template struct TS1<int, 10>; // expected-error{{template argument for template type parameter must be a type}}

template <typename, typename ...> struct TS2 {}; // expected-note{{template is declared here}}
template struct TS2<>; // expected-error{{too few template arguments for class template 'TS2'}}
template struct TS2<int>;
template struct TS2<int, int>;

template <typename = int, typename ...> struct TS3 {}; // expected-note{{template parameter is declared here}}
template struct TS3<>; // expected-note{{previous explicit instantiation is here}}
template struct TS3<int>; // expected-error{{duplicate explicit instantiation of 'TS3}}
template struct TS3<int, int>;
template struct TS3<10>; // expected-error{{template argument for template type parameter must be a type}}
