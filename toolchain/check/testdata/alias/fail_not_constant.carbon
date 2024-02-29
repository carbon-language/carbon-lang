// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  var a: i32 = 0;
  var b: i32* = &a;
  // CHECK:STDERR: fail_not_constant.carbon:[[@LINE+3]]:13: ERROR: Alias initializer must be a name reference.
  // CHECK:STDERR:   alias c = *b;
  // CHECK:STDERR:             ^~
  alias c = *b;
}

// CHECK:STDOUT: --- fail_not_constant.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.2: type = ptr_type i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc8: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %a.var, %.loc8
// CHECK:STDOUT:   %.loc9_13: type = ptr_type i32 [template = constants.%.2]
// CHECK:STDOUT:   %b.var: ref i32* = var b
// CHECK:STDOUT:   %b: ref i32* = bind_name b, %b.var
// CHECK:STDOUT:   %a.ref: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc9_17: i32* = addr_of %a.ref
// CHECK:STDOUT:   assign %b.var, %.loc9_17
// CHECK:STDOUT:   %b.ref: ref i32* = name_ref b, %b
// CHECK:STDOUT:   %.loc13_14: i32* = bind_value %b.ref
// CHECK:STDOUT:   %.loc13_13: ref i32 = deref %.loc13_14
// CHECK:STDOUT:   %c: <error> = bind_alias c, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
