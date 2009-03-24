// RUN: clang-cc -fsyntax-only -verify %s
int aa(int b, int x[sizeof b]) {}

void foo(int i, int A[i]) {}

