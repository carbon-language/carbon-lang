// RUN: %clang_cc1 -emit-llvm -o - -fexceptions -triple x86_64-apple-darwin10 %s | FileCheck %s

struct X {
  X();
  ~X();
};

struct Y { };

// CHECK: define void @_Z1fv
void f() {
  // CHECK: call i32 @__cxa_guard_acquire(i64* @_ZGVZ1fvE1x)
  // CHECK: invoke void @_ZN1XC1Ev
  // CHECK: call void @__cxa_guard_release(i64* @_ZGVZ1fvE1x)
  // CHECK: call i32 @__cxa_atexit
  // CHECK: br
  static X x;
  // CHECK: call void @__cxa_guard_abort(i64* @_ZGVZ1fvE1x)
  // CHECK: call void @__cxa_rethrow() noreturn
  // CHECK: unreachable

  // CHECK: call i8* @__cxa_allocate_exception
  throw Y();
}
