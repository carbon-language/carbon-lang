#include <stdio.h>

static int c;

struct A {
  A() { ++c; }
  A(const A&) { ++c; }
  ~A() { --c; }
};

struct B {
  A a;
  B() { A a; throw 1; }
};

int main() {
  try {
    B b;
  } catch (...) {}
  if (!c) printf("All ok!\n");
  return c;
}

