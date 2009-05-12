// RUN: clang-cc -fsyntax-only -verify %s
template<typename T> int foo(T), bar(T, T); // expected-error{{single entity}}
