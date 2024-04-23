// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(T:! type) -> T*;

fn F(T:! type) -> T* {
  return F(T);
}

// CHECK:STDOUT: --- redeclare.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.loc7: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %T.loc7_6.1: type = param T
// CHECK:STDOUT:     %T.loc7_6.2: type = bind_symbolic_name T, %T.loc7_6.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.ref.loc7: type = name_ref T, %T.loc7_6.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc7: type = ptr_type T [symbolic = constants.%.1]
// CHECK:STDOUT:     %return.var.loc7: ref T* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.loc9: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %T.loc9_6.1: type = param T
// CHECK:STDOUT:     @F.%T: type = bind_symbolic_name T, %T.loc9_6.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.ref.loc9: type = name_ref T, @F.%T [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc9: type = ptr_type T [symbolic = constants.%.1]
// CHECK:STDOUT:     @F.%return: ref T* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%T: type) -> T* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%F.loc7 [template = file.%F.loc7]
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %F.call: init T* = call %F.ref(%T.ref)
// CHECK:STDOUT:   %.loc10_14.1: T* = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc10_14.2: T* = converted %F.call, %.loc10_14.1
// CHECK:STDOUT:   return %.loc10_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
