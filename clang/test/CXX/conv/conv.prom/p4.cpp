// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

enum X : short { A, B };
extern decltype(+A) x;
extern int x;

enum Y : long { C, D };
extern decltype(+C) y;
extern long y;
