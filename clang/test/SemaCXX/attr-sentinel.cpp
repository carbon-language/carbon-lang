// RUN: clang-cc -fsyntax-only -verify %s
void f(int, ...) __attribute__((sentinel));

void g() {
  f(1, 2, __null);
}
