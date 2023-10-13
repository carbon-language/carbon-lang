// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  // CHECK:STDERR: fail_name_not_found.carbon:[[@LINE+3]]:16: ERROR: Name `a` not found.
  // CHECK:STDERR:   var b: i32 = a[0];
  // CHECK:STDERR:                ^
  var b: i32 = a[0];
}

// CHECK:STDOUT: file "fail_name_not_found.carbon" {
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b: ref i32 = var "b"
// CHECK:STDOUT:   %a.ref: <error> = name_reference "a", <error>
// CHECK:STDOUT:   %.loc11: i32 = int_literal 0
// CHECK:STDOUT:   assign %b, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
