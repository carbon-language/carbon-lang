// RUN: clang-cc -fsyntax-only -verify %s

// C++-specific tests for integral constant expressions.

const int c = 10;
int ar[c];
