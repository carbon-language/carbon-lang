// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wnewline-eof -verify %s

// The following line isn't terminated, don't fix it.
void foo() {}