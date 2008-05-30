// RUN: clang -fsyntax-only -verify %s
static int f = 10;
static int b = f; // expected-error {{initializer element is not constant}}

float r  = (float) &r; // expected-error {{initializer element is not constant}}
long long s = (long long) &s;
_Bool t = &t;
