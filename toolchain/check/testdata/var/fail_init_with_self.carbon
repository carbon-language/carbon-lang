// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  // CHECK:STDERR: fail_init_with_self.carbon:[[@LINE+3]]:16: ERROR: Name `x` not found.
  // CHECK:STDERR:   var x: i32 = x;
  // CHECK:STDERR:                ^
  var x: i32 = x;
}

// CHECK:STDOUT: file "fail_init_with_self.carbon" {
// CHECK:STDOUT:   %Main = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x: ref i32 = var "x"
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
