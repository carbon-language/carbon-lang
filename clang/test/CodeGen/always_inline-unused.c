// Test alwaysinline definitions w/o any non-direct-call uses.
// None of the declarations are emitted. Stub are only emitted when the original
// function can not be discarded.

// RUN: %clang_cc1 -disable-llvm-optzns -emit-llvm %s -o - | FileCheck %s

void __attribute__((__always_inline__)) f1() {}
inline void __attribute__((__always_inline__)) f2() {}
static inline void __attribute__((__always_inline__)) f3() {}
inline void __attribute__((gnu_inline, __always_inline__)) f4() {}
static inline void __attribute__((gnu_inline, __always_inline__)) f5() {}
inline void __attribute__((visibility("hidden"), __always_inline__)) f6() {}
inline void __attribute__((visibility("hidden"), gnu_inline, __always_inline__)) f7() {}

void g() {
  f1();
  f2();
  f3();
  f4();
  f5();
  f6();
  f7();
}

// CHECK: define void @f1()
// CHECK-NOT: void @f2()
// CHECK-NOT: void @f3()
// CHECK: define void @f4()
// CHECK-NOT: void @f5()
// CHECK-NOT: void @f6()
// CHECK: define hidden void @f7()
