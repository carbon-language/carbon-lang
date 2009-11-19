// RUN: clang-cc -fsyntax-only -verify %s

template<int N>
void f() {
  int a[] = { 1, 2, 3, N };
  unsigned numAs = sizeof(a) / sizeof(int);
}

template void f<17>();

