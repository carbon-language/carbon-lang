// RUN: %clang_cc1 -fsyntax-only -verify %s

#include "cuda.h"

__global__ void g1(int x) {}

template <typename T> void t1(T arg) {
  g1<<<arg, arg>>>(1);
}

void h1(int x) {}
int h2(int x) { return 1; }

int main(void) {
  g1<<<1, 1>>>(42);

  t1(1);

  h1<<<1, 1>>>(42); // expected-error {{kernel call to non-global function h1}}

  int (*fp)(int) = h2;
  fp<<<1, 1>>>(42); // expected-error {{must have void return type}}
}
