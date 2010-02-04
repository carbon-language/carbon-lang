// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 -O3 | FileCheck %s

namespace {

static int counter;
  
struct A {
  A() : i(0) { counter++; }
  ~A() { counter--; }
  int i;
};

void g(int) { }

void f1(bool b) {
  g(b ? A().i : 0);
  g(b || A().i);
  g(b && A().i);
}

struct Checker {
  Checker() {
    f1(true);
    f1(false);
  }
};

Checker c;

}

// CHECK: define i32 @_Z10getCounterv()
int getCounter() {
  // CHECK: ret i32 0
  return counter;
}
