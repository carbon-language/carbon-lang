// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_type_mismatch_assignment.carbon" {

fn Main() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a: ref i32 = var "a"
  var a: i32 = 3;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]: i32 = int_literal 3
  // CHECK:STDOUT:   assign %a, %.loc[[@LINE-2]]
  // CHECK:STDERR: fail_type_mismatch_assignment.carbon:[[@LINE+3]]:5: Cannot implicitly convert from `f64` to `i32`.
  // CHECK:STDERR:   a = 5.6;
  // CHECK:STDERR:     ^
  a = 5.6;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]: f64 = real_literal 56e-1
  // CHECK:STDOUT:   assign %a, <error>
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
