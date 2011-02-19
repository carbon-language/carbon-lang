// RUN: %clang_cc1 -analyze -analyzer-check-objc-mem -verify %s

int f1(char *dst) {
  char *p = dst + 4;
  char *q = dst + 3;
  return !(q >= p);
}

long f2(char *c) {
  return long(c) & 1;
}

bool f3() {
  return !false;
}

namespace {

struct A { };
struct B {
  operator A() { return A(); }
};

A f(char *dst) {
  B b;
  return b;
}

}
