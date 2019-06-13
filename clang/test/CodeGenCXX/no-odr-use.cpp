// RUN: %clang_cc1 -emit-llvm -o - -triple x86_64-linux-gnu %s | FileCheck %s

// CHECK: @__const._Z1fi.a = private unnamed_addr constant {{.*}} { i32 1, [2 x i32] [i32 2, i32 3], [3 x i32] [i32 4, i32 5, i32 6] }

struct A { int x, y[2]; int arr[3]; };
// CHECK-LABEL: define i32 @_Z1fi(
int f(int i) {
  // CHECK: call void {{.*}}memcpy{{.*}}({{.*}}, {{.*}} @__const._Z1fi.a
  constexpr A a = {1, 2, 3, 4, 5, 6};

  // CHECK-LABEL: define {{.*}}@"_ZZ1fiENK3$_0clEiM1Ai"(
  return [] (int n, int A::*p) {
    // CHECK: br i1
    return (n >= 0
      // CHECK: getelementptr inbounds [3 x i32], [3 x i32]* getelementptr inbounds ({{.*}} @__const._Z1fi.a, i32 0, i32 2), i64 0, i64 %
      ? a.arr[n]
      // CHECK: br i1
      : (n == -1
        // CHECK: getelementptr inbounds i8, i8* bitcast ({{.*}} @__const._Z1fi.a to i8*), i64 %
        // CHECK: bitcast i8* %{{.*}} to i32*
        // CHECK: load i32
        ? a.*p
        // CHECK: getelementptr inbounds [2 x i32], [2 x i32]* getelementptr inbounds ({{.*}} @__const._Z1fi.a, i32 0, i32 1), i64 0, i64 %
        // CHECK: load i32
        : a.y[2 - n]));
  }(i, &A::x);
}
