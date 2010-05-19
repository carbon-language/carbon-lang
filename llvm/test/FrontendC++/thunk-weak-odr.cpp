// RUN: %llvmgxx %s -S -o - | FileCheck %s
// <rdar://problem/7929157>

struct A {
  virtual int f() { return 1; }
};

struct B {
  virtual int f() { return 2; }
};

struct C : A, B {
  virtual int f() { return 3; }
};

struct D : C {
  virtual int f() { return 4; }
};

static int f(D* d) {
  B* b = d;
  return b->f();
};

int g() {
  D d;
  return f(&d);
}

// Thunks should be marked as "weak ODR", not just "weak".
//
// CHECK: define weak_odr i32 @_ZThn8_N1C1fEv
// CHECK: define weak_odr i32 @_ZThn8_N1D1fEv
