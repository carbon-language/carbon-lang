// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  var x: i32 = 0;
}

// CHECK:STDOUT: file "decl_with_init.carbon" {
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x: ref i32 = var "x"
// CHECK:STDOUT:   %.loc8: i32 = int_literal 0
// CHECK:STDOUT:   assign %x, %.loc8
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
