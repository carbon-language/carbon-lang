// RUN: clang-cc -fsyntax-only  %s

void f(...) {
  int g(int(...));
}
