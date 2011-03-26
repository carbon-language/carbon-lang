// RUN: %clang_cc1 -analyze -analyzer-checker=core -verify %s

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

void *f4(int* w) {
  return reinterpret_cast<void*&>(w);
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

namespace {

struct S {
    void *p;
};

void *f(S* w) {
    return &reinterpret_cast<void*&>(*w);
}

}

namespace {

struct C { 
  void *p;
  static void f();
};

void C::f() { }

}
