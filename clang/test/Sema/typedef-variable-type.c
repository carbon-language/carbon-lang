// RUN: clang %s -verify -fsyntax-only -pedantic

typedef int (*a)[!.0]; // expected-error{{variable length array declared outside of any function}}
