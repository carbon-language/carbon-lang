// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x

template<typename ... Args = int> struct S1 { }; // expected-error{{template parameter pack cannot have a default argument}}
template<typename ... Args, typename T> struct S2 { }; // expected-error{{template parameter pack must be the last template parameter}}
