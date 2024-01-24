// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 {
  var n: i32 = 0;
  var p: i32* = &n;

  return *p;
}

// CHECK:STDOUT: --- basic.carbon
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
// CHECK:STDOUT:   %.loc9_13: type = ptr_type i32 [template = constants.%.2]
// CHECK:STDOUT:   %p.var: ref i32* = var p
// CHECK:STDOUT:   %p: ref i32* = bind_name p, %p.var
// CHECK:STDOUT:   %n.ref: ref i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc9_17: i32* = addr_of %n.ref
// CHECK:STDOUT:   assign %p.var, %.loc9_17
// CHECK:STDOUT:   %p.ref: ref i32* = name_ref p, %p
// CHECK:STDOUT:   %.loc11_11: i32* = bind_value %p.ref
// CHECK:STDOUT:   %.loc11_10.1: ref i32 = deref %.loc11_11
// CHECK:STDOUT:   %.loc11_10.2: i32 = bind_value %.loc11_10.1
// CHECK:STDOUT:   return %.loc11_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
