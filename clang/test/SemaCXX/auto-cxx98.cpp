// RUN: clang-cc -fsyntax-only -verify %s -std=c++98
void f() {
  auto int a;
  int auto b;
}
