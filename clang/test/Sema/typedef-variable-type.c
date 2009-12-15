// RUN: %clang_cc1 %s -verify -fsyntax-only -pedantic

typedef int (*a)[!.0]; // expected-warning{{size of static array must be an integer constant expression}}
