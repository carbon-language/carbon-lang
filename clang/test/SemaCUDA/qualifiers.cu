// RUN: %clang_cc1 -fsyntax-only -verify %s

#include "cuda.h"

__global__ void g1(int x) {}
