// RUN: %clang_cc1 -fsyntax-only -verify -Wvla %s

void test1(int n) {
  int v[n]; // expected-warning {{variable length array used}}
}

void test2(int n, int v[n]) { // expected-warning {{variable length array used}}
}

void test3(int n, int v[n]); // expected-warning {{variable length array used}}

template<typename T>
void test4(int n) {
  int v[n]; // expected-warning {{variable length array used}}
}

template<typename T>
void test5(int n, int v[n]) { // expected-warning {{variable length array used}}
}

template<typename T>
void test6(int n, int v[n]); // expected-warning {{variable length array used}}

template<typename T>
void test7(int n, T v[n]) { // expected-warning {{variable length array used}}
}

