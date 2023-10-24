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
// CHECK:STDOUT:   %y.loc14_5.1: ref i32 = var "y"
// CHECK:STDOUT:   %y.loc14_5.2: ref i32 = bind_name "y", %y.loc14_5.1
// CHECK:STDOUT:   %x.ref: <error> = name_reference "x", <error>
// CHECK:STDOUT:   assign %y.loc14_5.1, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.loc8_7.1: ref i32 = var "x"
// CHECK:STDOUT:   %x.loc8_7.2: ref i32 = bind_name "x", %x.loc8_7.1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
