// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Empty {}

fn F(T:! Empty) {
  // TODO: Support conversion from `T` to `type`.
  // CHECK:STDERR: fail_as_type_of_type.carbon:[[@LINE+3]]:10: ERROR: Cannot implicitly convert from `Empty` to `type`.
  // CHECK:STDERR:   var x: T;
  // CHECK:STDERR:          ^
  var x: T;
}

// CHECK:STDOUT: --- fail_as_type_of_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Empty [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %Empty.decl = interface_decl @Empty {} [template = constants.%.1]
// CHECK:STDOUT:   %F: <function> = fn_decl @F {
// CHECK:STDOUT:     %Empty.ref: type = name_ref Empty, %Empty.decl [template = constants.%.1]
// CHECK:STDOUT:     %T.loc9_6.1: Empty = param T
// CHECK:STDOUT:     @F.%T: Empty = bind_symbolic_name T, %T.loc9_6.1 [symbolic]
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%T: Empty) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %T.ref: Empty = name_ref T, %T [symbolic = %T]
// CHECK:STDOUT:   %x.var: ref <error> = var x
// CHECK:STDOUT:   %x: ref <error> = bind_name x, %x.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
