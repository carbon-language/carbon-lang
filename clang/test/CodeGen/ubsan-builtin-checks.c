// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -w -emit-llvm -o - %s -fsanitize=builtin | FileCheck %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -w -emit-llvm -o - %s -fsanitize=builtin | FileCheck %s --check-prefix=NOT-UB

// NOT-UB-NOT: __ubsan_handle_invalid_builtin

// CHECK: define{{.*}} void @check_ctz
void check_ctz(int n) {
  // CHECK: [[NOT_ZERO:%.*]] = icmp ne i32 [[N:%.*]], 0, !nosanitize
  // CHECK-NEXT: br i1 [[NOT_ZERO]]
  //
  // Handler block:
  // CHECK: call void @__ubsan_handle_invalid_builtin
  // CHECK-NEXT: unreachable
  //
  // Continuation block:
  // CHECK: call i32 @llvm.cttz.i32(i32 [[N]], i1 true)
  __builtin_ctz(n);

  // CHECK: call void @__ubsan_handle_invalid_builtin
  __builtin_ctzl(n);

  // CHECK: call void @__ubsan_handle_invalid_builtin
  __builtin_ctzll(n);
}

// CHECK: define{{.*}} void @check_clz
void check_clz(int n) {
  // CHECK: [[NOT_ZERO:%.*]] = icmp ne i32 [[N:%.*]], 0, !nosanitize
  // CHECK-NEXT: br i1 [[NOT_ZERO]]
  //
  // Handler block:
  // CHECK: call void @__ubsan_handle_invalid_builtin
  // CHECK-NEXT: unreachable
  //
  // Continuation block:
  // CHECK: call i32 @llvm.ctlz.i32(i32 [[N]], i1 true)
  __builtin_clz(n);

  // CHECK: call void @__ubsan_handle_invalid_builtin
  __builtin_clzl(n);

  // CHECK: call void @__ubsan_handle_invalid_builtin
  __builtin_clzll(n);
}
