// RUN: %clang_cc1 -triple x86_64 %s -fno-use-cxa-atexit -emit-llvm -o - | FileCheck %s

// CHECK: define internal void @_GLOBAL__D_a()
// CHECK:   call void @_ZN1AD1Ev(%class.A* @b)
// CHECK:   call void @_ZN1AD1Ev(%class.A* @a)
// CHECK: }

class A {
public:
  A();
  ~A();
};

A a, b;
