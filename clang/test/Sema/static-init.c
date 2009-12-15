// RUN: %clang_cc1 -triple i386-pc-linux-gnu -fsyntax-only -verify %s

#include <stdint.h>

static int f = 10;
static int b = f; // expected-error {{initializer element is not a compile-time constant}}

float r  = (float) (intptr_t) &r; // expected-error {{initializer element is not a compile-time constant}}
intptr_t s = (intptr_t) &s;
_Bool t = &t;


union bar {
  int i;
};

struct foo {
  unsigned ptr;
};

union bar u[1];
struct foo x = {(intptr_t) u}; // no-error
struct foo y = {(char) u}; // expected-error {{initializer element is not a compile-time constant}}
