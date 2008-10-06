// RUN: clang -fsyntax-only  %s

void f(...) {
  int g(int(...));
}
