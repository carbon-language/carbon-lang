// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 -O3 | FileCheck %s

namespace {

static int ctorcalls;
static int dtorcalls;
  
struct A {
  A() : i(0) { ctorcalls++; }
  ~A() { dtorcalls++; }
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

// CHECK: define i32 @_Z12getCtorCallsv()
int getCtorCalls() {
  // CHECK: ret i32 3
  return ctorcalls;
}

// CHECK: define i32 @_Z12getDtorCallsv()
int getDtorCalls() {
  // CHECK: ret i32 3
  return dtorcalls;
}

// CHECK: define zeroext i1 @_Z7successv()
bool success() {
  // CHECK: ret i1 true
  return ctorcalls == dtorcalls;
}
