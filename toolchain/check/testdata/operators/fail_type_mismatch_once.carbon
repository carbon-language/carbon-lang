// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_type_mismatch_once.carbon" {

fn Main() -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> i32 {
// CHECK:STDOUT: !entry:
  // The following line has two mismatches, but after the first, it shouldn't
  // keep erroring.
  // CHECK:STDERR: fail_type_mismatch_once.carbon:[[@LINE+3]]:13: Cannot implicitly convert from `i32` to `f64`.
  // CHECK:STDERR:   return 12 + 3.4 + 12;
  // CHECK:STDERR:             ^
  return 12 + 3.4 + 12;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_10: i32 = int_literal 12
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_15: f64 = real_literal 34e-1
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_13: <error> = add <error>, %.loc[[@LINE-3]]_15
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_21: i32 = int_literal 12
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_19: <error> = add <error>, %.loc[[@LINE-5]]_21
  // CHECK:STDOUT:   return <error>
  // CHECK:STDOUT: }
}
