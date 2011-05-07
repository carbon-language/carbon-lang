// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

struct A {
  A() { x = 10; }
  int x;
};

const A x;

// CHECK: @_ZL1x = internal global

struct X {
  int (*fp)(int, int);
};

int add(int x, int y) { return x + y; }

// CHECK: @x2 = constant
extern const X x2;
const X x2 = { &add };
