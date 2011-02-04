// RUN: %clang_cc1 -fsyntax-only -verify

// PR9137
void f0(int x) : {};
void f1(int x) try {};
