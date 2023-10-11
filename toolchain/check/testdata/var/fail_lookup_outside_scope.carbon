// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  var x: i32;
}

// CHECK:STDERR: fail_lookup_outside_scope.carbon:[[@LINE+3]]:14: ERROR: Name `x` not found.
// CHECK:STDERR: var y: i32 = x;
// CHECK:STDERR:              ^
var y: i32 = x;

// CHECK:STDOUT: file "fail_lookup_outside_scope.carbon" {
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT:   %y: ref i32 = var "y"
// CHECK:STDOUT:   %x.ref: <error> = name_reference "x", <error>
// CHECK:STDOUT:   assign %y, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x: ref i32 = var "x"
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
