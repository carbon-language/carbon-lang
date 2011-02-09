// RUN: %clang_cc1 -fsyntax-only -verify %s

#include "cuda.h"

__global__ void g1(int x) {}

template <typename T> void t1(T arg) {
  g1<<<arg, arg>>>(1);
}

int main(void) {
  g1<<<1, 1>>>(42);

  t1(1);
}
