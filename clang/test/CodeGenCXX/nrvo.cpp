// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fexceptions -o - %s | FileCheck --check-prefix=CHECK-EH %s

// Test code generation for the named return value optimization.
class X {
public:
  X();
  X(const X&);
  ~X();
};

// CHECK: define void @_Z5test0v
// CHECK-EH: define void @_Z5test0v
X test0() {
  X x;
  // CHECK-NOT: call void @_ZN1XD1Ev
  // CHECK: ret void
  // CHECK-EH: ret void
  return x;
  // CHECK-EH: ehcleanup:
  // CHECK-EH: invoke void @_ZN1XD1Ev
}

// CHECK: define void @_Z5test1b(
X test1(bool B) {
  // CHECK: call void @_ZN1XC1Ev
  X x;
  // CHECK-NOT: call void @_ZN1XD1Ev
  // CHECK: ret void
  if (B)
    return (x);
  return x;
  // CHECK-EH: ehcleanup:
  // CHECK-EH: invoke void @_ZN1XD1Ev
}

// CHECK: define void @_Z5test2b
// CHECK-EH: define void @_Z5test2b
X test2(bool B) {
  // No NRVO
  // CHECK: call void @_ZN1XC1Ev
  X x;
  // CHECK: call void @_ZN1XC1Ev
  X y;
  // CHECK: call void @_ZN1XC1ERKS_
  // CHECK-EH: invoke void @_ZN1XC1ERKS_
  if (B)
    return y;
  // CHECK: call void @_ZN1XC1ERKS_
  // CHECK-EH: invoke void @_ZN1XC1ERKS_
  return x;
  // CHECK: call void @_ZN1XD1Ev
  // CHECK: call void @_ZN1XD1Ev
  // CHECK: ret void
  // CHECK-EH: ehcleanup:
  // CHECK-EH: invoke void @_ZN1XD1Ev
  // CHECK-EH: invoke void @_ZN1XD1Ev
}

X test3(bool B) {
  // FIXME: We don't manage to apply NRVO here, although we could.
  {
    X y;
    return y;
  }
  X x;
  return x;
}
