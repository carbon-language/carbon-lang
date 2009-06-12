// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x

template<typename... Args = int> struct S { }; // expected-error{{template parameter pack cannot have a default argument}}
