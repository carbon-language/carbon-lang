// RUN: %llvmgxx %s -S -emit-llvm -O0 -o - | FileCheck %s
// <rdar://problem/7929157>

// Thunks should be marked as "ODR".

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

// CHECK: define weak_odr {{.*}} @_ZThn8_N1D1fEv
