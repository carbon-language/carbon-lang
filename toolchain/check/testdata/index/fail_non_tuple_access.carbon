// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  // CHECK:STDERR: fail_non_tuple_access.carbon:[[@LINE+3]]:6: ERROR: `i32` does not support indexing.
  // CHECK:STDERR:   0[1];
  // CHECK:STDERR:      ^
  0[1];
}

// CHECK:STDOUT: file "fail_non_tuple_access.carbon" {
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_3: i32 = int_literal 0
// CHECK:STDOUT:   %.loc11_5: i32 = int_literal 1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
