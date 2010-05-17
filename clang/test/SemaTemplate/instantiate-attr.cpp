// RUN: %clang_cc1 -fsyntax-only -verify %s
template <typename T>
struct A {
  char a __attribute__((aligned(16)));

  struct B {
    typedef T __attribute__((aligned(16))) i16;
    i16 x;
  };
};
int a[sizeof(A<int>) == 16 ? 1 : -1];
int a2[sizeof(A<int>::B) == 16 ? 1 : -1];

