// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  // CHECK:STDERR: fail_storage_is_literal.carbon:[[@LINE+3]]:10: ERROR: Cannot implicitly convert from `i32` to `type`.
  // CHECK:STDERR:   var x: 1 = 1;
  // CHECK:STDERR:          ^
  var x: 1 = 1;
}

// CHECK:STDOUT: file "fail_storage_is_literal.carbon" {
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_10: i32 = int_literal 1
// CHECK:STDOUT:   %x: ref <error> = var "x"
// CHECK:STDOUT:   %.loc11_14: i32 = int_literal 1
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
