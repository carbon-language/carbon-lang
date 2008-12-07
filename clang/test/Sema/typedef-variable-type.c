// RUN: clang %s -verify -fsyntax-only -pedantic

typedef int (*a)[!.0]; // expected-error{{variably modified type declaration not allowed at file scope}}
