// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "binary_op.carbon" {

fn Main() -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> i32 {
// CHECK:STDOUT: !entry:
  return 12 + 34;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_10: i32 = int_literal 12
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_15: i32 = int_literal 34
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_13: i32 = add %.loc[[@LINE-3]]_10, %.loc[[@LINE-3]]_15
  // CHECK:STDOUT:   return %.loc[[@LINE-4]]_13
  // CHECK:STDOUT: }
}
