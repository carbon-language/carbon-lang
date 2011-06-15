// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

// rdar://problem/9158302
// This should not use a memmove_collectable in non-GC mode.
namespace test0 {
  struct A {
    id x;
  };

  // CHECK:    define [[A:%.*]]* @_ZN5test04testENS_1AE(
  // CHECK:      alloca
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: store
  // CHECK-NEXT: call noalias i8* @_Znwm(
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @llvm.memset.p0i8.i64(
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(
  // CHECK-NEXT: ret
  A *test(A a) {
    return new A(a);
  }
}

