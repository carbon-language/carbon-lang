// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR6076
void f();
void (&g)() = (void(), f);

int a[1];
int (&b)[1] = (void(), a);
