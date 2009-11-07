// RUN: clang-cc -fsyntax-only -verify %s
template <typename T>
struct A {
  char a __attribute__((aligned(16)));
};
int a[sizeof(A<int>) == 16 ? 1 : -1];

