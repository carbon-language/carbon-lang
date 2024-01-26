// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(T:! type) {
  var p: T*;
  let n: T = *p;
}

// CHECK:STDOUT: --- type_param.carbon
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
// CHECK:STDOUT: fn @F(%T: type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %T.ref.loc8: type = name_ref T, %T [symbolic = %T]
// CHECK:STDOUT:   %.loc8: type = ptr_type T [symbolic = constants.%.1]
// CHECK:STDOUT:   %p.var: ref T* = var p
// CHECK:STDOUT:   %p: ref T* = bind_name p, %p.var
// CHECK:STDOUT:   %T.ref.loc9: type = name_ref T, %T [symbolic = %T]
// CHECK:STDOUT:   %p.ref: ref T* = name_ref p, %p
// CHECK:STDOUT:   %.loc9_15: T* = bind_value %p.ref
// CHECK:STDOUT:   %.loc9_14.1: ref T = deref %.loc9_15
// CHECK:STDOUT:   %.loc9_14.2: T = bind_value %.loc9_14.1
// CHECK:STDOUT:   %n: T = bind_name n, %.loc9_14.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
