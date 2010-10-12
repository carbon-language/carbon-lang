// RUN: %clang_cc1 -fsyntax-only -verify %s 

void choice(int);
int choice(bool);

void test() {
  // Result of ! must be type bool.
  int i = choice(!1);
}

// rdar://8018252
void f0() {
  extern void f0_1(int*);
  register int x;
  f0_1(&x);
}

template <class T> void bar(T &x) { T::fail(); }
template <class T> void bar(volatile T &x) {}
void f1() {
  volatile int x;
  bar(x = 5);
}
