// RUN: %clang  -emit-llvm -g -S %s -o - | FileCheck %s

// TAG_member is used to encode debug info for 'z' in A.
// CHECK: TAG_member
class A {
public:
  int z;
};

A *foo (A* x) {
  A *a = new A(*x);
  return a;
}

// Verify that we're not emitting a full definition of B in limit debug mode.
// RUN: %clang -emit-llvm -g -flimit-debug-info -S %s -o - | FileCheck %s
// CHECK-NOT: TAG_member

class B {
public:
  int y;
};

extern int bar(B *b);
int baz(B *b) {
  return bar(b);
}

