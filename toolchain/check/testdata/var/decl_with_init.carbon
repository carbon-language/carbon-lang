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
// CHECK:STDOUT:   %x.loc8_7.1: ref i32 = var "x"
// CHECK:STDOUT:   %x.loc8_7.2: ref i32 = bind_name "x", %x.loc8_7.1
// CHECK:STDOUT:   %.loc8: i32 = int_literal 0
// CHECK:STDOUT:   assign %x.loc8_7.1, %.loc8
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
