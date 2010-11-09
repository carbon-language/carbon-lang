// RUN: %clang_cc1 -std=c++98 -fsyntax-only -verify %s
void f(void *);
void g() { f(__nullptr); }
