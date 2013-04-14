// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s

int g();

// CHECK: @_ZZ1fvE1n = internal thread_local global i32 0
// CHECK: @_ZGVZ1fvE1n = internal thread_local global i8 0

// CHECK: @_ZZ8tls_dtorvE1s = internal thread_local global
// CHECK: @_ZGVZ8tls_dtorvE1s = internal thread_local global i8 0

// CHECK: @_ZZ8tls_dtorvE1t = internal thread_local global
// CHECK: @_ZGVZ8tls_dtorvE1t = internal thread_local global i8 0

// CHECK: @_ZZ8tls_dtorvE1u = internal thread_local global
// CHECK: @_ZGVZ8tls_dtorvE1u = internal thread_local global i8 0
// CHECK: @_ZGRZ8tls_dtorvE1u = internal thread_local global

// CHECK: define i32 @_Z1fv()
int f() {
  // CHECK: %[[GUARD:.*]] = load i8* @_ZGVZ1fvE1n, align 1
  // CHECK: %[[NEED_INIT:.*]] = icmp eq i8 %[[GUARD]], 0
  // CHECK: br i1 %[[NEED_INIT]]

  // CHECK: %[[CALL:.*]] = call i32 @_Z1gv()
  // CHECK: store i32 %[[CALL]], i32* @_ZZ1fvE1n, align 4
  // CHECK: store i8 1, i8* @_ZGVZ1fvE1n
  // CHECK: br label
  static thread_local int n = g();

  // CHECK: load i32* @_ZZ1fvE1n, align 4
  return n;
}

struct S { S(); ~S(); };
struct T { ~T(); };

// CHECK: define void @_Z8tls_dtorv()
void tls_dtor() {
  // CHECK: load i8* @_ZGVZ8tls_dtorvE1s
  // CHECK: call void @_ZN1SC1Ev(%struct.S* @_ZZ8tls_dtorvE1s)
  // CHECK: call i32 @__cxa_thread_atexit({{.*}}@_ZN1SD1Ev {{.*}} @_ZZ8tls_dtorvE1s{{.*}} @__dso_handle
  // CHECK: store i8 1, i8* @_ZGVZ8tls_dtorvE1s
  static thread_local S s;

  // CHECK: load i8* @_ZGVZ8tls_dtorvE1t
  // CHECK-NOT: _ZN1T
  // CHECK: call i32 @__cxa_thread_atexit({{.*}}@_ZN1TD1Ev {{.*}}@_ZZ8tls_dtorvE1t{{.*}} @__dso_handle
  // CHECK: store i8 1, i8* @_ZGVZ8tls_dtorvE1t
  static thread_local T t;

  // CHECK: load i8* @_ZGVZ8tls_dtorvE1u
  // CHECK: call void @_ZN1SC1Ev(%struct.S* @_ZGRZ8tls_dtorvE1u)
  // CHECK: call i32 @__cxa_thread_atexit({{.*}}@_ZN1SD1Ev {{.*}} @_ZGRZ8tls_dtorvE1u{{.*}} @__dso_handle
  // CHECK: store i8 1, i8* @_ZGVZ8tls_dtorvE1u
  static thread_local const S &u = S();
}
