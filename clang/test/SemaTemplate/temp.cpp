// RUN: %clang_cc1 -fsyntax-only -verify %s


// p3
template<typename T> int foo(T), bar(T, T); // expected-error{{single entity}}
