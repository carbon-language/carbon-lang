// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_non_tuple_access.carbon" {

fn Main() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
  // CHECK:STDERR: fail_non_tuple_access.carbon:[[@LINE+3]]:6: `i32` does not support indexing.
  // CHECK:STDERR:   0[1];
  // CHECK:STDERR:      ^
  0[1];
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_3: i32 = int_literal 0
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_5: i32 = int_literal 1
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
