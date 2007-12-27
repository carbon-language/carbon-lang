// RUN: clang -fsyntax-only -verify %s

int @"s" = 5;  // expected-error {{unknown}}
