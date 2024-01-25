// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  let T:! type = i32;
  var p: T*;
  var a: T = *p;
}

// CHECK:STDOUT: --- generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = ptr_type T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, i32 [symbolic]
// CHECK:STDOUT:   %T.ref.loc9: type = name_ref T, %T [symbolic = %T]
// CHECK:STDOUT:   %.loc9: type = ptr_type T [symbolic = constants.%.1]
// CHECK:STDOUT:   %p.var: ref T* = var p
// CHECK:STDOUT:   %p: ref T* = bind_name p, %p.var
// CHECK:STDOUT:   %T.ref.loc10: type = name_ref T, %T [symbolic = %T]
// CHECK:STDOUT:   %a.var: ref T = var a
// CHECK:STDOUT:   %a: ref T = bind_name a, %a.var
// CHECK:STDOUT:   %p.ref: ref T* = name_ref p, %p
// CHECK:STDOUT:   %.loc10_15: T* = bind_value %p.ref
// CHECK:STDOUT:   %.loc10_14.1: ref T = deref %.loc10_15
// CHECK:STDOUT:   %.loc10_14.2: T = bind_value %.loc10_14.1
// CHECK:STDOUT:   assign %a.var, %.loc10_14.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
