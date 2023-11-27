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
// CHECK:STDOUT:   %x.var: ref i32 = var x
// CHECK:STDOUT:   %x: ref i32 = bind_name x, %x.var
// CHECK:STDOUT:   %.loc7: i32 = int_literal 0
// CHECK:STDOUT:   assign %x.var, %.loc7
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %y.var: ref i32 = var y
// CHECK:STDOUT:   %y: ref i32 = bind_name y, %y.var
// CHECK:STDOUT:   %x.ref: ref i32 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc10: i32 = bind_value %x.ref
// CHECK:STDOUT:   assign %y.var, %.loc10
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
