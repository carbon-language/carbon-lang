// RUN: clang-cc -fsyntax-only -verify %s

template<typename T> typedef T X; // expected-error{{typedef cannot be a template}}
