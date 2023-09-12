// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_storage_is_literal.carbon" {

fn Main() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
  // CHECK:STDERR: fail_storage_is_literal.carbon:[[@LINE+3]]:10: Cannot implicitly convert from `i32` to `type`.
  // CHECK:STDERR:   var x: 1 = 1;
  // CHECK:STDERR:          ^
  var x: 1 = 1;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_10: i32 = int_literal 1
  // CHECK:STDOUT:   %x: ref <error> = var "x"
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_14: i32 = int_literal 1
  // CHECK:STDOUT:   assign %x, <error>
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
