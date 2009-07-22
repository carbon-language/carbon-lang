// RUN: clang-cc -fsyntax-only -verify %s
void aa(int b, int x[sizeof b]) {}

void foo(int i, int A[i]) {}

