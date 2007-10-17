// RUN: clang -fsyntax-only -verify %s
static int f = 10;
static int b = f; // expected-error {{initializer element is not constant}}
