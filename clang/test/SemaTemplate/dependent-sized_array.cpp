// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s

template<int N>
void f() {
  int a[] = { 1, 2, 3, N };
  unsigned numAs = sizeof(a) / sizeof(int);
}

template void f<17>();


template<int N>
void f1() {
  int a0[] = {}; // expected-warning{{zero}}
  int a1[] = { 1, 2, 3, N };
  int a3[sizeof(a1)/sizeof(int) != 4? 1 : -1]; // expected-error{{negative}}
}
