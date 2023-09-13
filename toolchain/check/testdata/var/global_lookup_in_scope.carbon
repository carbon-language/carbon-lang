// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: i32 = 0;

fn Main() {
  var y: i32 = x;
}

// CHECK:STDOUT: file "global_lookup_in_scope.carbon" {
// CHECK:STDOUT:   %x: ref i32 = var "x"
// CHECK:STDOUT:   %.loc7: i32 = int_literal 0
// CHECK:STDOUT:   assign %x, %.loc7
// CHECK:STDOUT:   %.loc9 = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %y: ref i32 = var "y"
// CHECK:STDOUT:   %.loc7: i32 = bind_value package.%x
// CHECK:STDOUT:   assign %y, %.loc7
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
