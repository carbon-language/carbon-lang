// RUN: clang %s -fsyntax-only
// PR1892
void f(double a[restrict][5]);  // should promote to restrict ptr.
void f(double (* restrict a)[5]);

