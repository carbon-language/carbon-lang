// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR5548
struct A {~A();};
void a(const A* x) {
  x->~A();
}
