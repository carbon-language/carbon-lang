// RUN: clang-cc -fsyntax-only -std=c99 -verify %s
void bb(int sz, int ar[sz][sz]) { }
