// RUN: clang-cc -fsyntax-only -std=c99 -verify %s
int bb(int sz, int ar[sz][sz]) { }
