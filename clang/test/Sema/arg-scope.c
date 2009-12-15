// RUN: %clang_cc1 -fsyntax-only -verify %s
void aa(int b, int x[sizeof b]) {}

void foo(int i, int A[i]) {}

