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

namespace test1 {
  template <class T> void bar(T &x) { T::fail(); }
  template <class T> void bar(volatile T &x) {}

  void test_ints() {
    volatile int x;
    bar(x = 5);
    bar(x += 5);
  }

  enum E { E_zero };
  void test_enums() {
    volatile E x;
    bar(x = E_zero);
    bar(x += E_zero); // expected-error {{incompatible type}}
  }
}
