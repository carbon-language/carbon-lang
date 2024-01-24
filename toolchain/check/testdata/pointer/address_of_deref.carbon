// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 {
  var n: i32 = 0;
  return *&*&n;
}

// CHECK:STDOUT: --- address_of_deref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.2: type = ptr_type i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.var: ref i32 = var n
// CHECK:STDOUT:   %n: ref i32 = bind_name n, %n.var
// CHECK:STDOUT:   %.loc8: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %n.var, %.loc8
// CHECK:STDOUT:   %n.ref: ref i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc9_13: i32* = addr_of %n.ref
// CHECK:STDOUT:   %.loc9_12: ref i32 = deref %.loc9_13
// CHECK:STDOUT:   %.loc9_11: i32* = addr_of %.loc9_12
// CHECK:STDOUT:   %.loc9_10.1: ref i32 = deref %.loc9_11
// CHECK:STDOUT:   %.loc9_10.2: i32 = bind_value %.loc9_10.1
// CHECK:STDOUT:   return %.loc9_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
